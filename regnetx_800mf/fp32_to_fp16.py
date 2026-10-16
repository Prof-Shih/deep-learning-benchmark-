import onnx
from onnxconverter_common import float16

INPUT_MODEL = "RegNetX-800MF_dds_8gpu.onnx"
OUTPUT_MODEL = "RegNetX-800MF_dds_8gpu_fp16.onnx"

model = onnx.load(INPUT_MODEL)

fp16_model = float16.convert_float_to_float16(
    model,
    keep_io_types=True
)

onnx.save(fp16_model, OUTPUT_MODEL)

print(f"Saved FP16 model: {OUTPUT_MODEL}")