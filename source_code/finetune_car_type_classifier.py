from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
from datasets import load_dataset,ClassLabel,load_from_disk
import shutil
import gc
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_x = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#transform image
def transform_fn(example):
    example['image'] = transform_x(example['image'])
    example["label"] = torch.tensor(example["label"])
    return example

def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    return torch.stack(images), torch.tensor(labels)

def save_model_state(model, epoch, train_loss_list, path="model_checkpoint.pth"):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder,exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss_list
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

if __name__=="__main__":

    # Load preprocessed huggingface dataset and inser dataset into torch data loader
    ds = load_from_disk("./data/Stanford-Cars")
    transformed_ds = ds.with_transform(transform_fn)

    train_loader = DataLoader(transformed_ds["train"], batch_size=32, shuffle=True,collate_fn=collate_fn)
    eval_loader = DataLoader(transformed_ds["test"], batch_size=32,collate_fn=collate_fn)

    # Load MobileNetV3-Large pretrained on ImageNet
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # Modify the final layer for a custom number of classes (10)
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=len(ds['train'].features['label'].names))

    ####### TRAIN #########

# Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.001)
    mobilenet_v3_large = mobilenet_v3_large.to(device)
    # Training loop
    num_epochs = 30
    train_loss = []
    mobilenet_v3_large.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = mobilenet_v3_large(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_loss.append(running_loss/len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # save result
    save_model_state(mobilenet_v3_large, epoch, train_loss, path=f"./output/mobilenet_v3_large_checkpoint_{epoch+1}.pth")