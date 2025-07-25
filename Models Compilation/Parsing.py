# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

chosen_hw_arch = "hailo8l" # For Hailo 15 devices, use 'hailo15h' # For Mini PCIe modules or Hailo8R devices, use 'hailo8r'

onnx_path = "./ONNX_models/convnextv2_atto.onnx"
onnx_model_name = "convnextv2_atto"

runner = ClientRunner(hw_arch=chosen_hw_arch)

hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    start_node_names=["input"],
    #end_node_names=["192"],
    net_input_shapes={"input": [1,3,384,384]},
)

hailo_model_har_name = f"{onnx_model_name}_hailo_model_convnextv2_atto.har"
runner.save_har(hailo_model_har_name)