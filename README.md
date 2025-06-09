Car Retrieval System
=====================
**By:** *Musa Izzanardi Wijanarko*

This project is intended for completing technical test. This Car Retrieval System will be able to detect multiple car
instances then classify/retrieve what is the type (eg. MPV, Sedan, Hatchback, etc) of the detected car.

This system consists of two models:
- Car Detection model, using [X]
- Car Body Type Classifier model, using [Y]

---


# Getting started

This guide is for UNIX-based OS. 

## Requrements

- Python>=3.10.2
- All dependencies are in the `requirements.txt`
- Ideally a linux-based system, equipped with CUDA (check it with `nvcc --version`)

## Setting up

To set this system up, you need to:

1. Clone the repo with `git clone`
2. go to the inside of the directory with `cd car-retrieval system`
3. (Optional) create a virtual environment `.env` with `python3 -m venv .env`
4. Activate it with `source .env/bin/activate`
5. Install the required dependencies `pip install -r requirements.txt`

## Information

---
# Car Detection Model

This object detection model was based on [DETR-ResNet50](microsoft/conditional-detr-resnet-50) with 40M+ parameters, and was trained on the Indian Driving Dataset [(Varma et al, 2018)](https://arxiv.org/pdf/1811.10200v1) 40K used for object detection task. We will use this dataset since it is resemble the Indonesian road condition.

The dataset has 9 labels and defined as such:
```json
{
    "0": "traffic sign",
    "1": "motorcycle",
    "2": "car",
    "3": "rider",
    "4": "person",
    "5": "truck",
    "6": "autorickshaw",
    "7": "vehicle fallback",
    "8": "bus"
}
```
And we manually convert this dataset into to a CPPE-5-like (YOLO annotation) format and uploaded it into huggingface.\
🤗 [**Huggingface Dataset Link**](https://huggingface.co/datasets/izzako/IDD_Detection_CPPE5)

---
# Car Classifier Model

The classifier model was trained on the Standford Cars dataset introduced by [Krause et al. (2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.html). This dataset does not contains car type, however, [mayurmahukar](https://github.com/mayurmahurkar/Stanford-Cars-Body-Data?utm_source=chatgpt.com) create a mapping to 10 types of car, which are:
```python
{
    0: 'Coupe',
    1: 'Sedan',
    2: 'Cab',
    3: 'Convertible',
    4: 'SUV',
    5: 'Minivan',
    6: 'Hatchback',
    7: 'Other',
    8: 'Van',
    9: 'Wagon'
 }
```
