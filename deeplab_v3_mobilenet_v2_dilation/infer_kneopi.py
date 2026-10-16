import os
import sys
import time
import statistics

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
import cv2
import psutil
import numpy as np

# =========================
# 固定設定
# =========================
USB_PORT_ID = 5
SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
IMAGE_DIR_PATH = os.path.join(PWD, '../Test')

# Hailo Model Zoo deeplab_v3_mobilenet_v2 + NCHW wrapper
# NEF input: [1, 3, 513, 513]
INPUT_WIDTH = 513
INPUT_HEIGHT = 513

# 你這顆 NEF 是 NCHW wrapper，所以 CPU 端直接準備 NCHW normalized tensor
INPUT_SHAPE_NCHW = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)

# ImageNet normalize
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Generic Data Inference 輸入 buffer 格式
# 一般 NCHW wrapper 若是吃 float input，維持 float32。
# 若你的 ioinfo.csv / NEF descriptor 顯示 input 是 int8 fixed-point，再改成 "int8"。
INPUT_BUFFER_MODE = "float32"   # "float32" or "int8"
INPUT_RADIX = None             # INPUT_BUFFER_MODE="int8" 時可填 radix；None 則嘗試從 NEF descriptor 讀取

# 幾張印一次進度
PRINT_EVERY = 10

# 溫度檔
TEMP_PATH = '/sys/class/thermal/thermal_zone0/temp'


# =========================
# 工具函式
# =========================
def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            files.append(path)
    files.sort()
    return files


def read_temp_c():
    try:
        with open(TEMP_PATH, 'r') as f:
            raw = f.read().strip()
        t = float(raw)
        if t > 1000:
            t = t / 1000.0
        return t
    except Exception:
        return None


def _safe_shape_to_list(shape_obj):
    if shape_obj is None:
        return None

    # list / tuple / numpy array
    if isinstance(shape_obj, (list, tuple, np.ndarray)):
        try:
            return [int(x) for x in list(shape_obj)]
        except Exception:
            return list(shape_obj)

    # Kneron shape object sometimes has attributes or is printable only
    for attr in ('shape', 'dims', 'data'):
        if hasattr(shape_obj, attr):
            try:
                value = getattr(shape_obj, attr)
                if isinstance(value, (list, tuple, np.ndarray)):
                    return [int(x) for x in list(value)]
            except Exception:
                pass

    try:
        return list(shape_obj)
    except Exception:
        return str(shape_obj)


def _get_nodes_from_model(model_desc, node_kind):
    """
    Try common Kneron descriptor field names.
    node_kind: "input" or "output"
    """
    if node_kind == "input":
        candidates = (
            'input_nodes',
            'input_node_list',
            'input_node_descriptor_list',
            'inputs',
            'input_node',
        )
    else:
        candidates = (
            'output_nodes',
            'output_node_list',
            'output_node_descriptor_list',
            'outputs',
            'output_node',
        )

    for name in candidates:
        if hasattr(model_desc, name):
            try:
                nodes = getattr(model_desc, name)
                if nodes is None:
                    continue
                if isinstance(nodes, (list, tuple)):
                    return list(nodes)
                return [nodes]
            except Exception:
                pass

    return []


def _get_node_name(node, default_name):
    for attr in ('name', 'node_name', 'tensor_name'):
        if hasattr(node, attr):
            try:
                value = getattr(node, attr)
                if value is not None:
                    return str(value)
            except Exception:
                pass
    return default_name


def _get_node_shape(node):
    # 優先印 ONNX/原始 shape，再印 NPU shape
    for attr in ('shape_onnx', 'shape', 'shape_npu', 'dimensions', 'dims'):
        if hasattr(node, attr):
            try:
                value = getattr(node, attr)
                shape = _safe_shape_to_list(value)
                if shape is not None:
                    return shape
            except Exception:
                pass
    return None


def print_nef_io_shape(model_nef_descriptor):
    print('[NEF Input/Output Shape]')

    try:
        model_desc = model_nef_descriptor.models[0]
    except Exception:
        print(' - Cannot parse model descriptor. Full NEF descriptor:')
        print(model_nef_descriptor)
        return

    print(f' - Model ID: {getattr(model_desc, "id", "unknown")}')

    input_nodes = _get_nodes_from_model(model_desc, 'input')
    output_nodes = _get_nodes_from_model(model_desc, 'output')

    if not input_nodes and not output_nodes:
        print(' - Cannot directly parse input/output nodes from descriptor.')
        print(' - Full NEF descriptor:')
        print(model_nef_descriptor)
        return

    if input_nodes:
        print(' - Inputs:')
        for i, node in enumerate(input_nodes):
            name = _get_node_name(node, f'input_{i}')
            shape = _get_node_shape(node)
            print(f'   [{i}] name={name}, shape={shape}')
    else:
        print(' - Inputs: cannot parse from descriptor')

    if output_nodes:
        print(' - Outputs:')
        for i, node in enumerate(output_nodes):
            name = _get_node_name(node, f'output_{i}')
            shape = _get_node_shape(node)
            print(f'   [{i}] name={name}, shape={shape}')
    else:
        print(' - Outputs: cannot parse from descriptor')


def try_get_input_radix(model_nef_descriptor):
    """
    如果要用 int8 fixed-point input，嘗試從 NEF descriptor 抓 input radix。
    抓不到就回傳 None。
    """
    try:
        model_desc = model_nef_descriptor.models[0]
        input_nodes = _get_nodes_from_model(model_desc, 'input')
        if not input_nodes:
            return None

        node = input_nodes[0]

        # 常見欄位：node.quantization_parameters.quantized_fixed_point_descriptor_list[0].radix
        qparams = getattr(node, 'quantization_parameters', None)
        if qparams is not None:
            qlist = getattr(qparams, 'quantized_fixed_point_descriptor_list', None)
            if qlist:
                radix = getattr(qlist[0], 'radix', None)
                if radix is not None:
                    return int(radix)

        # 其他可能欄位
        for attr in ('radix', 'input_radix'):
            if hasattr(node, attr):
                value = getattr(node, attr)
                if value is not None:
                    return int(value)
    except Exception:
        pass

    return None


def preprocess_deeplab_mobilenetv2_nchw(image_path, width, height):
    """
    Hailo deeplab_v3_mobilenet_v2 常見前處理：
    BGR(OpenCV read) -> RGB -> resize 513x513 -> /255 -> ImageNet normalize -> NCHW
    回傳 shape = [1, 3, 513, 513], dtype=float32
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f'Cannot read image: {image_path}')

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img_chw = np.transpose(img, (2, 0, 1))
    img_nchw = np.expand_dims(img_chw, axis=0)
    img_nchw = np.ascontiguousarray(img_nchw, dtype=np.float32)

    if img_nchw.shape != INPUT_SHAPE_NCHW:
        raise ValueError(f'Unexpected input shape: {img_nchw.shape}, expected: {INPUT_SHAPE_NCHW}')

    return img_nchw


def make_input_buffer(img_nchw, model_nef_descriptor=None):
    """
    GenericDataInferenceDescriptor 的 GenericInputNodeData 需要 bytes buffer。
    預設 float32 NCHW；若你的 NEF 實際 input 是 fixed-point int8，可切 INPUT_BUFFER_MODE="int8"。
    """
    if INPUT_BUFFER_MODE.lower() == "float32":
        return np.ascontiguousarray(img_nchw.astype(np.float32)).tobytes()

    if INPUT_BUFFER_MODE.lower() == "int8":
        radix = INPUT_RADIX
        if radix is None and model_nef_descriptor is not None:
            radix = try_get_input_radix(model_nef_descriptor)

        if radix is None:
            raise ValueError(
                'INPUT_BUFFER_MODE="int8" but INPUT_RADIX is None and cannot be parsed from NEF descriptor.'
            )

        scale = float(1 << int(radix))
        q = np.round(img_nchw * scale)
        q = np.clip(q, -128, 127).astype(np.int8)
        return np.ascontiguousarray(q).tobytes()

    raise ValueError(f'Unsupported INPUT_BUFFER_MODE: {INPUT_BUFFER_MODE}')


def build_generic_data_descriptor(model_id, input_buffer):
    """
    使用 Generic Data Inference，因為前處理已經在 CPU 端完成。
    """
    generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model_id,
        inference_number=0
    )

    generic_inference_input_descriptor.input_node_data_list = [
        kp.GenericInputNodeData(buffer=input_buffer)
    ]

    return generic_inference_input_descriptor


def retrieve_float_outputs(generic_result):
    outputs = []
    num_output_node = generic_result.header.num_output_node

    for node_idx in range(num_output_node):
        node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=generic_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
        )
        outputs.append(node_output)

    return outputs


def _get_output_ndarray(node_output):
    arr = getattr(node_output, 'ndarray', None)
    if arr is None and isinstance(node_output, dict):
        arr = node_output.get('ndarray', None)
    if arr is None:
        raise ValueError('Cannot get ndarray from output node')
    return np.asarray(arr)


def _get_output_shape(node_output):
    shape = getattr(node_output, 'shape', None)
    if shape is None and isinstance(node_output, dict):
        shape = node_output.get('shape', None)

    if shape is not None:
        shape = _safe_shape_to_list(shape)
        return shape

    arr = _get_output_ndarray(node_output)
    return list(arr.shape)


def decode_segmentation_map_from_output(output_node):
    """
    依照實際 output shape 判斷 segmentation map：
    - [1, C, H, W] -> argmax axis=1
    - [1, H, W, C] -> argmax axis=3
    - [C, H, W] -> argmax axis=0
    - [H, W, C] -> argmax axis=2
    - [1, H, W] / [H, W] -> already class map
    - flattened -> 嘗試依 513x513 還原
    """
    out = _get_output_ndarray(output_node)
    out = np.asarray(out)

    # 移除很單純的 batch 維度，保留後面判斷也可以
    if out.ndim == 4:
        # [1, C, H, W]
        if out.shape[0] == 1 and out.shape[1] > 1 and out.shape[2] > 1 and out.shape[3] > 1:
            seg = np.argmax(out, axis=1)[0]
            return seg.astype(np.uint8)

        # [1, H, W, C]
        if out.shape[0] == 1 and out.shape[-1] > 1 and out.shape[1] > 1 and out.shape[2] > 1:
            seg = np.argmax(out, axis=3)[0]
            return seg.astype(np.uint8)

    if out.ndim == 3:
        # [1, H, W] already map
        if out.shape[0] == 1 and out.shape[1] > 1 and out.shape[2] > 1:
            return out[0].astype(np.uint8)

        # [C, H, W]
        if out.shape[0] <= 256 and out.shape[1] > 1 and out.shape[2] > 1:
            seg = np.argmax(out, axis=0)
            return seg.astype(np.uint8)

        # [H, W, C]
        if out.shape[-1] <= 256 and out.shape[0] > 1 and out.shape[1] > 1:
            seg = np.argmax(out, axis=2)
            return seg.astype(np.uint8)

    if out.ndim == 2:
        return out.astype(np.uint8)

    if out.ndim == 1:
        hw = INPUT_HEIGHT * INPUT_WIDTH

        # already flattened map
        if out.size == hw:
            return out.reshape(INPUT_HEIGHT, INPUT_WIDTH).astype(np.uint8)

        # flattened logits: [C, H, W]
        if out.size % hw == 0:
            c = out.size // hw
            logits = out.reshape(1, c, INPUT_HEIGHT, INPUT_WIDTH)
            seg = np.argmax(logits, axis=1)[0]
            return seg.astype(np.uint8)

    raise ValueError(f'Cannot infer segmentation map from output shape: {out.shape}')


def print_first_output_info_and_segmap(generic_result):
    print('[Runtime Output Shape]')
    outputs = retrieve_float_outputs(generic_result)

    if len(outputs) == 0:
        print(' - No output node retrieved.')
        return

    for i, out in enumerate(outputs):
        shape = _get_output_shape(out)
        name = getattr(out, 'name', None)
        if name is None and isinstance(out, dict):
            name = out.get('name', None)
        if name is None:
            name = f'output_{i}'
        print(f' - Output[{i}] name={name}, shape={shape}')

    # DeepLab segmentation 通常只會用第一個 output
    seg_map = decode_segmentation_map_from_output(outputs[0])
    print(f' - Detected segmentation map shape: {list(seg_map.shape)}')
    print(f' - Segmentation classes in this sample: {np.unique(seg_map).tolist()}')


# =========================
# 主程式
# =========================
if __name__ == '__main__':
    print('[Check Device USB Speed]')
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=USB_PORT_ID)
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != usb_speed and \
           kp.UsbSpeed.KP_USB_SPEED_HIGH != usb_speed:
            print('\033[91m[Warning] Device is not running at super/high speed.\033[0m')
        else:
            print(' - Success')
    except Exception as exception:
        print(f'Error: check device USB speed fail, port ID = {USB_PORT_ID}, error = [{exception}]')
        sys.exit(0)

    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[USB_PORT_ID])
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'Error: connect device fail, port ID = {USB_PORT_ID}, error = [{exception}]')
        sys.exit(0)

    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    print('[Upload Firmware]')
    try:
        kp.core.load_firmware_from_file(
            device_group=device_group,
            scpu_fw_path=SCPU_FW_PATH,
            ncpu_fw_path=""
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload firmware failed, error = '{exception}'")
        sys.exit(0)

    print('[Upload Model]')
    try:
        model_nef_descriptor = kp.core.load_model_from_file(
            device_group=device_group,
            file_path=MODEL_FILE_PATH
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload model failed, error = '{exception}'")
        sys.exit(0)

    # 先印 NEF input/output shape
    print_nef_io_shape(model_nef_descriptor)

    print('[Collect Images]')
    image_paths = get_image_paths(IMAGE_DIR_PATH)
    if len(image_paths) == 0:
        print(f'Error: no images found in {IMAGE_DIR_PATH}')
        sys.exit(0)

    total_images = len(image_paths)
    print(f' - Found {total_images} images')

    process = psutil.Process(os.getpid())
    temps = []
    ram_list = []
    pure_infer_times = []
    valid_count = 0
    skipped_count = 0

    runtime_output_printed = False

    print('[Starting Benchmark]')
    total_time_start = time.perf_counter()

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            img_nchw = preprocess_deeplab_mobilenetv2_nchw(
                image_path=image_path,
                width=INPUT_WIDTH,
                height=INPUT_HEIGHT
            )
            input_buffer = make_input_buffer(
                img_nchw=img_nchw,
                model_nef_descriptor=model_nef_descriptor
            )
        except Exception as exception:
            print(f'[Skip] {image_path} -> {exception}')
            skipped_count += 1
            continue

        generic_inference_input_descriptor = build_generic_data_descriptor(
            model_id=model_nef_descriptor.models[0].id,
            input_buffer=input_buffer
        )

        t0 = time.perf_counter()
        try:
            kp.inference.generic_data_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )
            generic_result = kp.inference.generic_data_inference_receive(
                device_group=device_group
            )
        except kp.ApiKPException as exception:
            print(f'[Skip] inference failed on {image_path}, error = {exception}')
            skipped_count += 1
            continue
        t1 = time.perf_counter()

        infer_time = t1 - t0
        pure_infer_times.append(infer_time)
        valid_count += 1

        # 只在第一張成功推論後印 output shape + 判斷 segmentation map
        # 不放進 t0~t1，避免影響原本「純推論時間」統計。
        if not runtime_output_printed:
            try:
                print_first_output_info_and_segmap(generic_result)
            except Exception as exception:
                print(f'[Warning] output retrieve / segmentation decode failed: {exception}')
            runtime_output_printed = True

        ram_mb = process.memory_info().rss / (1024 * 1024)
        ram_list.append(ram_mb)

        temp_c = read_temp_c()
        if temp_c is not None:
            temps.append(temp_c)

        if idx % PRINT_EVERY == 0 or idx == total_images:
            print(f' - Progress: {idx}/{total_images}')

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    pure_infer_time = sum(pure_infer_times)

    print('\n[Result]')
    print(f'Total images found: {total_images}')
    print(f'Valid images inferred: {valid_count}')
    print(f'Skipped images: {skipped_count}')

    if valid_count == 0:
        print('\nNo valid inference completed.')
        sys.exit(0)

    avg_infer_time = pure_infer_time / valid_count
    avg_fps = valid_count / pure_infer_time if pure_infer_time > 0 else 0.0
    overall_fps = valid_count / total_time if total_time > 0 else 0.0

    avg_ram = statistics.mean(ram_list) if ram_list else 0.0
    peak_ram = max(ram_list) if ram_list else 0.0

    if temps:
        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = statistics.mean(temps)
    else:
        min_temp = max_temp = avg_temp = 0.0

    print(f'\nTotal time: {total_time:.4f} s')
    print(f'Total pure inference time: {pure_infer_time:.4f} s')
    print(f'Average inference time: {avg_infer_time * 1000:.2f} ms')

    print(f'\nAverage FPS: {avg_fps:.2f}')
    print(f'Overall FPS: {overall_fps:.2f}')

    print(f'\nAverage RAM: {avg_ram:.2f} MB')
    print(f'Peak RAM: {peak_ram:.2f} MB')

    print(f'\nMin Temp: {min_temp:.2f} C')
    print(f'Max Temp: {max_temp:.2f} C')
    print(f'Average Temp: {avg_temp:.2f} C')
