from network import PPOAgent
import torch
import torch.utils.model_zoo as model_zoo
import utils
import onnxruntime
import numpy as np

torch_model = PPOAgent("onnx")

utils.load_weights(torch_model, "data/single_agent/player_1_5M.pth")
batch_size = 1

# set the model to inference mode
torch_model.eval()

x = torch.randn(35, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "pika.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
)

ort_session = onnxruntime.InferenceSession("pika.onnx", providers=["CPUExecutionProvider"])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
