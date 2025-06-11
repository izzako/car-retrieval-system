Car Retrieval System
=====================
**By:** *Musa Izzanardi Wijanarko*

This project is intended for completing technical test. This Car Retrieval System will be able to detect multiple car
instances then classify/retrieve what is the type (eg. MPV, Sedan, Hatchback, etc) of the detected car.

This system consists of two models:
- Car Detection model, using [DETR-ResNet50](microsoft/conditional-detr-resnet-50)
- Car Body Type Classifier model, using [MobilenetV3](https://docs.pytorch.org/vision/main/models/mobilenetv3.html)

The predicted result video can be seen on:
[▶ this link](https://drive.google.com/file/d/1xqp0DVImlBEv6S1S_StJM3ilmp_duxwj/view?usp=sharing)

You can also access the technical reports documentation [**here**](https://docs.google.com/document/d/1DH2EZPOcDfcGWRjLHZyOHD4DvVx-7_4VJbSbI6pMVbc/edit?usp=sharing) to see the system architecture, training documentation, etc.


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
3. (Optional) create a virtual environment `venv` with `python3 -m venv venv`
4. Activate it with `source venv/bin/activate`
5. Install the required dependencies `pip install -r requirements.txt`
6. You also need to setup a `.env` for these variables:
 - `HF_TOKEN`: your huggingface credential
 - `WANDB_PROJECT`: project name for tracking in Weight & Biases website
 - `HF_HOME`: huggingface default download path (models, datasets, etc)
 - `TORCH_HOME`: torch default download path
 - `VIDEO_GDRIVE_ID`: the google drive id of test video


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

The classifier model was based on [MobileNetV3](https://arxiv.org/pdf/1905.02244) available on [torch architecture](https://docs.pytorch.org/vision/main/models/mobilenetv3.html), and then trained on the Stanford Cars dataset introduced by [Krause et al. (2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.html). This dataset does not contains car type, however, [mayurmahukar](https://github.com/mayurmahurkar/Stanford-Cars-Body-Data?utm_source=chatgpt.com) create a mapping to 10 types of car, which are:
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
