#!/usr/bin/env python3

from model import *

import numpy as np
from dataclasses import dataclass

from torch import nn
from torch.onnx import _constants as torch_onnx_constants

import onnx
import onnxruntime
import tza

# Requires CUDA because of half precision (OIDN weights are stored in FP16 format), which is not fully supported (for this model?) on CPU.

# Prerequisites:
# - Python 3.8-3.11 (PyTorch currently supports only these versions)
# - CUDA 11.8 (has to match a PyTorch release), https://developer.nvidia.com/cuda-11-8-0-download-archive
# - PyTorch 2.0.1 with CUDA support, https://pytorch.org/get-started/locally/
# - OIDN, https://github.com/OpenImageDenoise/oidn

# Setup:
# 0. Copy this script to [OIDN_ROOT]/training
# 1. Install CUDA
# 2. Verify CUDA version in Terminal with 'nvcc --version', should match desired version of PyTorch release!
# 3. Setup virtual environment in Python: python -m venv MyPythonEnv
# 4. Activate python virtual evnironment: ./MyPythonEnv/Scripts/Activate.ps1
# 5. Install PyTorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 6. Install ONNX and ONNX runtime: pip3 install onnx onnxruntime

# Run:
# 1. Configure ModelParameter
# 2. Activate Python virtual environment: ./MyPythonEnv/Scripts/Activate.ps1
# 3. Run: Python .\training\onnx_export.py

# Collection of all parameters used to createa and export a version of the OIDN model with extracted weigths
@dataclass
class ModelParameter:
  use_albedo: bool = False      # use auxiliary feature buffer albeo
  use_normal: bool = False      # use auxiliary feature buffer albeo
  input_width: int = 1280       # input image width
  input_height: int = 720       # input image height
  input_dynamic: bool = False   # has input image width and height dynamic size?
  input_type: any = torch.half  # used type
  device: str = "cuda"          # device to use (note: dtype torch.half not supported on CPU for this model)
  hdr: bool = True              # hdr input


# Get the name of the weights file dependent on the used auxilary feature buffers
def get_weights_name(param: ModelParameter) -> str:
  parts = ['rt']
  parts.append('hdr' if param.hdr else 'ldr')
  if param.use_albedo: parts.append('alb')
  if param.use_normal: parts.append('nrm')
  return '_'.join(parts)


# Get the number of channels of the input tensor dependent on the used auxilary feature buffers
# # (with shape 1 x channels x height x with)
def get_num_channels(param: ModelParameter) -> int:
  return 3 + (3 if param.use_albedo else 0) + (3 if param.use_normal else 0)


# Use FP16? (recommended since weights are FP16)
def isFP16(param: ModelParameter) -> bool:
  return param.input_type == torch.half or param.input_type == torch.float16


# Create the torch model and load the weights into the model
def create(param: ModelParameter) -> nn.Module:
  torch.set_default_device(param.device)

  # Create the model
  torch_model = UNet(get_num_channels(param))
  if isFP16(param): torch_model.half()

  # Load the weights
  tza_full_filename = os.path.join('weights', get_weights_name(param) + '.tza')
  if os.path.exists(tza_full_filename):
    # Load weights from tensor archive (OIDN)
    weights_content = tza.Reader(tza_full_filename)

    # Create new state dict from the loaded weights
    loaded_state_dict = {}
    for tensor_key in torch_model.state_dict():
      loaded_state_dict[tensor_key] = torch.tensor(weights_content[tensor_key][0])
    
    # Load the new state dict
    torch_model.load_state_dict(loaded_state_dict)

    print('Loaded weights from ' + tza_full_filename)
  else:
    print('Could not load weights from ' + tza_full_filename)

  return torch_model


# Export a torch model to the ONNX format
def export(param: ModelParameter, torch_model: nn.Module) -> str:
  # Generate the filename for the model to export
  filename_parts = ['oidn2', get_weights_name(param) ]
  
  if param.input_dynamic: filename_parts.append("dyn")
  else: filename_parts.append(str(param.input_width) + 'x' + str(param.input_height))
  
  if isFP16(param): filename_parts.append('fp16')

  filename = '_'.join(filename_parts) + '.onnx'
  
  # Set the model to inference mode
  torch_model.eval()

  # Input to the model
  test_widht = param.input_width if param.input_width > 0 else 1280
  test_height = param.input_height if param.input_height > 0 else 720

  x = torch.randn(1, get_num_channels(param), test_height, test_widht, dtype=param.input_type, requires_grad=True)
  # torch_out = torch_model(x)

  dynamic_axes = None
  if param.input_dynamic:
    dynamic_axes = {'input' : {2 : 'height', 3: 'width'},    # variable length axes
                    'output' : {2 : 'height', 3: 'width'}}

  # Export the model
  torch.onnx.export(torch_model,                # model being run
                    x,                          # model input (or a tuple for multiple inputs)
                    filename,                   # where to save the model (can be a file or file-like object)
                    export_params=True,         # store the trained parameter weights inside the model file
                    opset_version=11,           # the ONNX version to export the model to
                    do_constant_folding=False,  # whether to execute constant folding for optimization
                    input_names=['input'],      # the model's input names
                    output_names=['output'],    # the model's output names
                    dynamic_axes=dynamic_axes   # variable length axes
  )

  print('Exported to ' + filename)
  return filename


# Test the exported ONNX model
def test(param: ModelParameter, torch_model: nn.Module, filename: str):
  # Verify the model's structure with the ONNX API
  onnx_model = onnx.load(filename)
  onnx.checker.check_model(onnx_model)

  # Generate reference output using torch
  # TODO match devices??
  test_widht = param.input_width if param.input_width > 0 else 1280
  test_height = param.input_height if param.input_height > 0 else 720

  x = torch.randn(1, get_num_channels(param), test_height, test_widht, dtype=param.input_type, requires_grad=True)
  torch_out = torch_model(x)

  # Run the model with ONNX Runtime
  ort_session = onnxruntime.InferenceSession(filename)

  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  # compute ONNX Runtime output prediction
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
  ort_outs = ort_session.run(None, ort_inputs)

  # compare ONNX Runtime and PyTorch results
  np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

  print("Exported model has been tested with ONNXRuntime, and the result looks good!")


# Helper to create, export and test a version of the OIDN model with extracted weights
def create_export_test(param: ModelParameter):
  torch_model = create(param)
  filename = export(param, torch_model)

  try:
    test(param, torch_model, filename)
  except AssertionError as assertion_error:
    print('Model test failed:', assertion_error)


if __name__ == '__main__':
  print(f'torch.onnx.ONNX_MIN_OPSET={torch_onnx_constants.ONNX_MIN_OPSET}, torch.onnx.ONNX_MAX_OPSET={torch_onnx_constants.ONNX_MAX_OPSET}')
  print(f'onnx.version={onnx.version.version!r}, opset={onnx.defs.onnx_opset_version()}, IR_VERSION={onnx.onnx_pb.IR_VERSION}')
  
  param = ModelParameter()

  # export model that denoises using only color buffer
  create_export_test(param)

  # export model that denoises using color and albedo buffers
  param.use_albedo = True
  create_export_test(param)

  # export model that denoises using all buffers (color, albedo and normal)
  param.use_normal = True
  create_export_test(param)

  ###

  # export everything again with dynamic axis width and height
  param.use_albedo = False
  param.use_normal = False
  param.input_dynamic = True

  # export model that denoises using only color buffer
  create_export_test(param)

  # export model that denoises using color and albedo buffers
  param.use_albedo = True
  create_export_test(param)

  # export model that denoises using all buffers (color, albedo and normal)
  param.use_normal = True
  create_export_test(param)