import onnx

in_path = "RegNetX-800MF_dds_8gpu.onnx"
out_path = "RegNetX-800MF_dds_8gpu_dynamic.onnx"

model = onnx.load(in_path)

for inp in model.graph.input:
    if inp.type.tensor_type.shape.dim:
        inp.type.tensor_type.shape.dim[0].dim_param = "batch"

for out in model.graph.output:
    if out.type.tensor_type.shape.dim:
        out.type.tensor_type.shape.dim[0].dim_param = "batch"

onnx.checker.check_model(model)
onnx.save(model, out_path)

print("Saved:", out_path)