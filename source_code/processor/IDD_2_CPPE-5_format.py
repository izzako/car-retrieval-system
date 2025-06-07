import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import json
from glob import glob

import xml.etree.ElementTree as ET
from datasets import Dataset, DatasetDict, Image
import shutil

id2label_idd = {
    0: 'traffic sign',
    1: 'motorcycle',
    2: 'car',
    3: 'rider',
    4: 'person',
    5: 'truck',
    6: 'autorickshaw',
    7: 'vehicle fallback',
    8: 'bus'
    
}
label2id_idd = {v: k for k, v in id2label_idd.items()}

json.dump(id2label_idd, open(os.path.join('data','IDD_Detection_CPPE5', 'id2label.json'), 'w'), indent=4)

#RESTRUCTURE DIRECTORY

DATA_DIR = os.path.join('data/IDD_Detection')

#create target dir
OUT_DIR = os.path.join('data','IDD_Detection_CPPE5')

for i in ['images','annotations']:
    for j in ['train','val']:
        os.makedirs(os.path.join(OUT_DIR,i,j),exist_ok=True)


#copy img and label in cityscape format from original dir to target dir
for j in ['train','val']:
    print(f'Processing {j} set...')
    with open(os.path.join(DATA_DIR,j+'.txt'), 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip('\n')
        if not line: continue
        src_img_path = os.path.join(DATA_DIR, 'JPEGImages', line+'.jpg')
        dst_img_path = os.path.join(OUT_DIR, 'images', j, '-'.join(line.split('/'))+'.jpg')
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        src_label_path = os.path.join(DATA_DIR, 'Annotations', line+'.xml')
        dst_label_path = os.path.join(DATA_DIR, 'annotations_xml', j, '-'.join(line.split('/'))+'.xml')

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

#convert xml to json

# Define paths
idd_ann_dir = os.path.join('.', 'data','IDD_Detection', 'annotations_xml')

# Helper: get all annotation files
xml_files = glob(os.path.join(idd_ann_dir,'val', '*.xml'))

image_id = 1
object_id = 1
def convert_to_cppe5_format(xml_file):
    """
    Convert XML annotations to COCO-CPPE5 format.
    Args:
        xml_file (str): Path to the XML file.
        label2id_idd (dict): Mapping from label names to IDs.
    Returns:
        dict: COCO-CPPE5 formatted annotations.
    """

    # Parse XML file
    if not os.path.exists(xml_file):
        print(f"File not found: {xml_file}")
        return {}
    global object_id
    global image_id
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = xml_file.split('/')[-1].replace('.xml', '')
    
    size = root.find('size')
    width = int(size.find('width').text) # type: ignore
    height = int(size.find('height').text) # type: ignore

    anns = []
    for obj in root.findall('object'):
        name = obj.find('name').text # type: ignore
        if name not in label2id_idd:
            continue
        category_id = label2id_idd[name]
        bndbox = obj.find('bndbox')
        try:
            xmin = int(float(bndbox.find('xmin').text))# type: ignore
            ymin = int(float(bndbox.find('ymin').text))# type: ignore
            xmax = int(float(bndbox.find('xmax').text))# type: ignore
            ymax = int(float(bndbox.find('ymax').text))# type: ignore
        except (AttributeError, ValueError, TypeError):
            continue
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        anns.append({
            "category_id": category_id,
            "area": w * h,
            "id": object_id,
            "bbox": [xmin, ymin, w, h]
        })
        object_id += 1

    if anns:
        categories = [ann['category_id'] for ann in anns]
        bboxes = [ann['bbox'] for ann in anns]
        idss = [ann['id'] for ann in anns]
        areas = [ann['area'] for ann in anns]
        label=    {
                "image_id": image_id,
                "filename"  : filename,
                "height": height,
                "width": width,
                "objects": {
                    "id": idss,
                    "category": categories,
                    "bbox": bboxes,
                    "area": areas
                }
            }
        
    else:
        categories = []
        bboxes = []
        idss = []
        areas = []
        label = {
            "image_id": image_id,
            "filename": filename,
            "height": height,
            "width": width,
            "objects": {
                "id": idss,
                "category": categories,
                "bbox": bboxes,
                "area": areas
            }
        }
    image_id += 1
    return label

for i in ['annotations']:
    for j in ['train','test','val']:
        os.makedirs(os.path.join(OUT_DIR,i,j),exist_ok=True)


for j in ['train','val']:
    xml_files = glob(os.path.join(idd_ann_dir,j, '*.xml'))
    print(f'Processing {j} set...')
    results = []
    for xml_file in tqdm(xml_files):
        result = convert_to_cppe5_format(xml_file)
        results.append(result)
    if results:
        output_file = os.path.join(OUT_DIR, 'annotations', f'{j}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)


# Upload to Hugging Face Hub
from datasets import load_dataset

ds = load_dataset("data/IDD_Detection_CPPE5",data_dir='data/IDD_Detection_CPPE5')
ds['validation'] = ds['validation'].filter(lambda example: len(example["objects"]['category'])!=0)
ds['train'] =  ds['train'].filter(lambda example: len(example["objects"]['category'])!=0)


ds.push_to_hub("IDD_Detection_CPPE5") #type:ignore