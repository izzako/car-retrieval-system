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

## Requrements

- Python 3.12.7
- MacOS Sequoaia 15.1

---
# Car Detection Model

This object detection model was trained on the Indian Driving Dataset [(Varma et al, 2018)](https://arxiv.org/pdf/1811.10200v1) that was initially used for image segmentation task. We will use this dataset since it is resemble Indonesian road condition.

---
# Car Classifier Model

The classifier model was trained on the CompCars dataset introduced by [Yang et al. (2015)](https://arxiv.org/pdf/1506.08959v2). This dataset contains up to 12 types of car, which are MPV, SUV, hatchback, sedan, minibus, fastback, estate, pickup, sports, crossover, convertible, and hardtop convertible, as shown as in the image below.
![dataset car type](images/image1.png)