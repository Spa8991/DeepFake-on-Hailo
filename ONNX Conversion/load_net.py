import torch
import timm

model_name= "mobilenetv4_conv_medium.e250_r384_in12k" #vit_tiny_patch16_384.augreg_in21k_ft_in1k #convnextv2_atto.fcmae_ft_in1k #fastvit_t8.apple_in1k #mobilenetv4_conv_medium.e250_r384_in12k
net = timm.create_model(model_name, pretrained=False)
if hasattr(net, 'classifier'):
        net.classifier = torch.nn.Linear(net.classifier.in_features,2)
else:
        if hasattr(net.head, 'fc'):
            net.head.fc = torch.nn.Linear(net.head.fc.in_features,2)
        else:
            net.head = torch.nn.Linear(net.head.in_features,2)
# Load the saved weights
checkpoint_path = "./pth_models/mobilenetv4_conv_medium.e250_r384_in12k999.pth"  # Update this with the actual path
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False).state_dict() # Load to CPU first      +++ weights_only=False


# Load the state_dict into the model
net.load_state_dict(state_dict)
net.eval()
dummy_input = torch.rand(1,3,384,384)
print (net(torch.rand(1,3,384,384)))

# Esportazione in ONNX
onnx_path = "./ONNX_models/mobilenetv4.onnx"
torch.onnx.export(
    net, 
    dummy_input, 
    onnx_path, 
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {1: "batch_size"}, "output": {0: "batch_size"}},  # Per batch dinamico
    opset_version=14  # Versione ONNX, puoi provare con 11 o 13 se ci sono errori
)

print(f"Modello convertito in ONNX e salvato come {onnx_path}")