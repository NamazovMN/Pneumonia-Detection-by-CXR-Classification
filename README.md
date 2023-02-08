# Pneumonia-Detection-by-CXR-Classification
Main objective of the project is to detect Pneumonia disease by Chest XRay (CXR) image classification. Two different Convolutional Neural Networks were developed in cascade fashion to classify images into Healthy and Pneumonia and Bacterial or Viral Pneumonia. More details can be found in project report.

## Table of Contents
* Brief Information about Project
* Applicational idea

* Evaluation and Results

## Brief Information about Project
In this project two different approaches were applied in order to detect Pneumonia disease by training on datasets of Chest X-Ray images. While the first model is used to detect the pneumonia disease,
the latter is used to detect the kind of it. Thus, we can call our model as Cascade CNN model for CXR classification.

## Idea
### First Approach
For the first approach, one Convolutional Neural Network was developed to classify CXR images into Viral-, Bacterial-Pneumonia and Healthy classes. Thus, it is called one model since, all classification is done explicitly. In other words, model handles classification task with images of 3 classes.
### Second Approach
The second approach works as 'divide and conquer' idea. It includes two CNN models. While the first one is used to classify CXR images into Pneumonia and Healthy classes (PH model), the latter is used for classifying pneumonia labelled images into Viral- and Bacterial-Pneumonia classes (VB model). Thus, we call the second approach as Cascade Model.
## Evaluation and Results
### First Approach
The first approach is simple mult-label classification task, so that evaluation is simple as its kinds. During training session we also evaluate the model over development dataset. Additionally, at each epoch F1-score is also computed. After the training, the model parameters with the best F1-score is used for generating the confusion matrix.
### Second Approach
For this model, we use PH model's dataset, in wich number of healthy and Pneumonia (we did not keep pneumonia sorts uniformed here) data are equal. Consufion matrix and F1 score is computed manually, since there is not such third-party function to compute F1 score or to generate confusion matrix for this kind of task. Evaluation is done in the following manner:
 * PH images are inferred by PH model and images are collected according to their actual label and prediction label;
 * Images which were classified as Pneumonia at the first step, are fed into VB model in order to detect the kind of Pneumonia disease. As we did at the first step, prediction results are collected according to their actual and prediction variable.
 * Using this information was collected through 2 steps, we generate confusion matrix and compute F1-score accordingly.
 
 <p align="center">
 <img src="np_4_loss.png" width="400" height="350">  <img src="np_4_acc.png" width="400" height="350">
 </p>

On the other hand you can see performance results of the second model of cascade system for 'Viral-Bacterial classification' task (VB-CNN) from the following images:

<p align="center">
 <img src="vb_4_loss.png" width="400" height="350">  <img src="vb_4_acc.png" width="400" height="350">
 </p>
 

The following image expresses the Confusion Matrix of Cascade Model through 3 classes:
<p align="center">
<img src="image.png" width="300" height="300">
 </p>
 
 
 
 ### Second Approach
 In the second approach, I directly collected results and evaluated results according to F1-score metrics. Following images express the train and test loss and accuracy results.
 
 The following images express loss and accuracy performance of the direct model, in which image classification is done for 3 classes:
 
<p align="center">
 <img src="om_4_loss.png" width="400" height="350">  <img src="om_4_acc.png" width="400" height="350">
 </p>
 

Additionally, confusion matrix is represented by the following image:

<p align="center">
<img src="DirectConfusion.png" width="300" height="300">
</p>

### Comparison
As a result, it is easy to see that both models performed very similarly that F1 scores are very close to each other. The following table expresses Micro and Macro averaged F1-scores of bot models' results.

<p align="center">
<img src="F1_scores.png" width="200" height="50">
</p>
