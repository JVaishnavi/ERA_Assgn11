# GRAD CAM application

The objective of this assignment is to use ResNet 18 on CIFAR10 dataset, predict the class of image, use GradCAM to see which part of the image is being used to predict the class.

Grad-CAM is a technique that utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score. The places where this gradient is large are exactly the places where the final score depends most on the data.

## Training and testing accuracy
![image](https://github.com/JVaishnavi/ERA_Assgn11/assets/11015405/651c3286-ae8d-4c58-b971-93f85d6f1c84)

## Misclassified image
![image](https://github.com/JVaishnavi/ERA_Assgn11/assets/11015405/acecc616-79b1-4179-8408-ab6d2a489311)

## GradCAM on misclassified image
![image](https://github.com/JVaishnavi/ERA_Assgn11/assets/11015405/f9d0dc5a-c491-4ae3-91ad-776016068805)

