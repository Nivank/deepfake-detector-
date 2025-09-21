import os
import torch
from torchvision.models import resnet50, efficientnet_b0
import timm

# Create a folder to store weights
os.makedirs("weights", exist_ok=True)

# -------- ResNet50 --------
print("Downloading ResNet50 weights...")
resnet_model = resnet50(pretrained=True)
torch.save(resnet_model.state_dict(), "weights/resnet50_weights.pth")
print("ResNet50 saved as weights/resnet50_weights.pth")

# -------- EfficientNet-B0 --------
print("Downloading EfficientNet-B0 weights...")
efficientnet_model = efficientnet_b0(pretrained=True)
torch.save(efficientnet_model.state_dict(), "weights/efficientnet_b0_weights.pth")
print("EfficientNet-B0 saved as weights/efficientnet_b0_weights.pth")

# -------- ViT-Base --------
print("Downloading ViT-Base weights...")
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
torch.save(vit_model.state_dict(), "weights/vit_base_weights.pth")
print("ViT-Base saved as weights/vit_base_weights.pth")

print("All weights downloaded and saved successfully!")
