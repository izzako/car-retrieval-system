from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import torch.optim as optim
from datasets import load_dataset, ClassLabel
import shutil
from torch.utils.data import DataLoader
from huggingface_hub import login
import torch.nn as nn


# login(token=os.environ['HF_TOKEN'])  # Replace with your Hugging Face token

url = "https://raw.githubusercontent.com/mayurmahurkar/Stanford-Cars-Body-Data/refs/heads/main/stanford_cars_type.csv"
df = pd.read_csv(url,index_col=0)
df = df.drop_duplicates(subset='car_code')

# code2label = {row['car_code']:row['car_type'] for _,row in df.iterrows()}
id2label = {i:name for i,name in enumerate(df['car_type'].unique())}
label2id = {name:i for i,name in id2label.items()}
code2id = {row['car_code']:label2id[row['car_type']] for _,row in df.iterrows()}

def map_code_to_label(example):
    example["label_type"] = code2id[example["label"]+1]
    return example


def convert_label_to_type(dataset):
    dataset = dataset.map(map_code_to_label)
    dataset = dataset.cast_column('label_type',ClassLabel(names=list(id2label.values())))
    dataset = dataset.remove_columns('label')
    dataset = dataset.rename_column('label_type','label')
    return dataset


def convert_grayscale_to_rgb(example):
    if example["image"].mode != "RGB":
        example["image"] = example["image"].convert("RGB")
    return example

if __name__ == "__main__":

    ds = load_dataset("Donghyun99/Stanford-Cars")

    print("convert car id to car body type,")
    ds['train'] = convert_label_to_type(ds['train'])
    ds['test'] = convert_label_to_type(ds['test'])

    print("convert grayscale format to rgb format for some of the data")
    ds['train'] = ds['train'].map(convert_grayscale_to_rgb,num_proc=1)
    ds['test'] = ds['test'].map(convert_grayscale_to_rgb,num_proc=1)

    ds.save_to_disk("data/Stanford-Cars") # save locally bcz the huggingface is down when i create this :(
