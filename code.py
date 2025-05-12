# Install required packages
!pip install minisom
!pip install roboflow ultralytics opencv-python albumentations torch torchvision

import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from minisom import MiniSom
from roboflow import Roboflow

# 1. Self-Organizing Map (SOM) for Anchor Selection
def compute_som_anchors(dataset_path, num_anchors=9):
    all_boxes = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    _, _, w, h = map(float, line.split()[4:])
                    all_boxes.append([w, h])
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}")
        return []

    all_boxes = np.array(all_boxes)
    if len(all_boxes) < num_anchors:
        print("Warning: Not enough boxes for SOM anchors.")
        return all_boxes

    som_grid_size = int(np.sqrt(num_anchors))
    som = MiniSom(som_grid_size, som_grid_size, 2, sigma=1.0, learning_rate=0.5)
    som.train_random(all_boxes, 10000)
    anchor_boxes = np.array([som.weights[x, y] for x, y in [som.winner(b) for b in all_boxes]])
    return anchor_boxes

# 2. Multi-Attention Fusion (MAF) with Channel-Spatial Attention
class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x) * x
        sa = self.spatial_att(torch.mean(ca, dim=1, keepdim=True)) * ca
        return sa

class MultiDimensionalAuxFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiDimensionalAuxFusion, self).__init__()
        self.conv_down = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_up = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.attention = ChannelSpatialAttention(in_channels)

    def forward(self, x):
        x_down = self.conv_down(x)
        x_up = self.conv_up(x_down)
        x_out = self.attention(x_up)
        return x_out

# 3. Weight Transfer Classification (WTC)
class WeightTransferClassification(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(WeightTransferClassification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.softmax(x)


class YOLOv8nEnhanced(YOLO):
    def __init__(self, model_path="yolov8n.pt", nc=4):
        super().__init__(model_path)
        model = self.model

        # Debug: Print the YOLO model structure
        print("\nInspecting YOLOv8 Model Structure:\n")
        for name, module in model.named_modules():
            print(f"Layer Name: {name} | Type: {type(module)}")

        # Apply MAF to a specific backbone layer (e.g., layer 4)
        if hasattr(model, "backbone"):
            for idx, layer in enumerate(model.backbone):
                if idx == 4:  # You can change this index based on inspection
                    if hasattr(layer, "conv"):
                        in_channels = layer.conv.in_channels
                    elif hasattr(layer[0], "conv"):
                        in_channels = layer[0].conv.in_channels
                    else:
                        raise ValueError("Cannot determine in_channels from layer.")

                    maf = MultiDimensionalAuxFusion(in_channels)
                    new_layer = nn.Sequential(layer, maf)
                    model.backbone[idx] = new_layer
                    break  # Inject MAF once for now

        # Find the detection layer with 'cv2' (typically YOLOv8 detection head)
        detect_layer = None
        detect_layer_name = None
        for name, module in model.named_modules():
            if hasattr(module, "cv2"):
                detect_layer = module
                detect_layer_name = name
                break

        if detect_layer is None:
            raise ValueError("Detection layer with attribute 'cv2' not found.")

        # Replace detection head with WTC-based layers
        if isinstance(detect_layer.cv2, nn.ModuleList):
            new_cv2 = nn.ModuleList()
            for layer in detect_layer.cv2:
                if isinstance(layer, nn.Sequential):
                    conv_layer = layer[0]
                    in_channels = conv_layer.conv.in_channels
                    out_channels = conv_layer.conv.out_channels
                    wtc = WeightTransferClassification(out_channels, nc)
                    new_cv2.append(wtc)
                else:
                    print("Unexpected layer format in cv2; skipping.")
                    new_cv2.append(layer)
            detect_layer.cv2 = new_cv2
        else:
            print("Detection layer 'cv2' is not a ModuleList. Structure may be incompatible.")

        # Set new number of classes
        detect_layer.nc = nc
        if detect_layer_name:
            model._modules[detect_layer_name] = detect_layer

        self.model = model


import os
from ultralytics import YOLO

# Dataset paths
dataset_path = "/kaggle/input/uav-china-spain"
data_yaml_path = os.path.join(dataset_path, "data.yaml")

if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"'data.yaml' not found at {data_yaml_path}")

print(f"Dataset Path: {dataset_path}")
print(f"data.yaml Path: {data_yaml_path}")

# Training and Evaluation
try:
    model = YOLOv8nEnhanced(model_path="yolov8m.pt", nc=4)
    model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        project="/kaggle/working/runs/detect",
        name="yolov8m_som_maf_wtc",
        exist_ok=True
    )
except Exception as e:
    print(f"Error during model training: {e}")

# Validation (val split)
try:
    results = model.val(
        data=data_yaml_path,
        imgsz=640
    )
    print(f"Validation Results: {results}")
except Exception as e:
    print(f"Error during model evaluation: {e}")

# Validation (test split)
try:
    results = model.val(
        data="/kaggle/input/uav-china-spain/data.yaml",
        split="test",
        save=True,
        plots=True
    )
    print(results)
except Exception as e:
    print(f"Validation Error: {e}")
