# Performance Analysis of Deep Learning Models in Driver Behavior Recognition

## Overview

This project investigates the effectiveness of various deep learning models in recognizing driver behavior to enhance road safety. The primary objective is to classify driver behaviors that could lead to accidents, such as drowsiness and distractions, and develop an alarm system to alert drivers in real-time. 

## Abstract

Road transportation is critical for the movement of goods and people, making road safety paramount. Despite advancements in vehicle safety features, driver errors remain a significant cause of accidents. This project addresses this issue by performing a comparative analysis of traditional deep learning and transfer learning models to classify driver behaviors using two datasets: the Driver Drowsiness Dataset (DDD) and the State Farm Distracted Driver Dataset. The goal is to develop an alarm system that sounds when unsafe driving behavior is detected.

## Datasets

1. **Driver Drowsiness Dataset (DDD) 2021**:
   - Contains over 40,000 cropped face images of drivers from videos.
   - Divided into two classes: drowsy and non-drowsy.

2. **State Farm Distracted Driver Dataset 2016**:
   - Contains over 20,000 images of drivers captured through dash cameras.
   - Spanning 10 classes representing different driver behaviors, including safe driving, texting, talking on the phone, operating the radio, drinking, and more.

## Methodology

### Data Preprocessing
- **Extracting region of interest**: Important eye data was extracted from frontal face data using HAAR cascade classifier.
- **Image Resizing**: All images were resized to a uniform size of 80x80 pixels.
- **Color Conversion**: Images were converted to a standard color format for consistency.
- **Image Enhancement**: Eye images extracted were enhanced by applying kernel sharpening filters and apply contrast and brightness transformations to the image.

### Model Training

Three deep learning models were used for comparative analysis:

1. **Traditional Convolutional Neural Network (CNN)**
2. **ResNet50 (Transfer Learning)**
3. **VGG16 (Transfer Learning)**

Each model was trained individually on both datasets to classify driver behavior.

### Alarm System Development

An ensemble of all trained models was used to develop an alarm system. This system averages the predictions from each model and sounds an alarm if unsafe driving behavior is detected.

## Results

- **Model Accuracy**: Each model achieved high accuracy on both datasets, with slight variations in performance.
- **Alarm System**: Successfully developed an alarm system that alerts drivers based on a threshold of predicted unsafe behavior.

## Potential Use Cases

1. **Real-Time Driver Monitoring**: Implementing the alarm system in vehicles to monitor driver behavior and alert them in case of drowsiness or distractions.
2. **Insurance**: Providing data to insurance companies for assessing driver behavior and offering personalized insurance rates.
3. **Regulatory Compliance**: Helping transport companies comply with safety regulations by monitoring and improving driver behavior.

## Conclusion

This project demonstrates the potential of deep learning models in enhancing road safety by accurately classifying driver behavior and developing a real-time alarm system. Implementing such systems can significantly reduce road accidents caused by driver errors.

## Authors

- [Rameshwari Kapoor](https://github.com/RameshwariKapoor)
- [Rashmi Kuliyal](https://github.com/Kuliyalrashmi)
- Nilesh Bhanot

## Note

- Model files for the GUI can be generated using model_distraction.ipynb/model_drowsiness.ipynb files or pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1HFW4kh8GnZYWHCBSU4JRAOX-BdoZTjFf?usp=sharing)
---

Feel free to customize further based on specific needs or additional details from your research.
