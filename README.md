# Project A Code - Ifham Ahmed 
Semantic Segmentation Using Adversarial Networks - Project A

Author: Ifham Ahmed

Institution: Monash University Malaysia

This repository contains the entire source code for Project A of my final year project.
The entire code was written in Python with the Keras deep learning framework and other helper libraries.

Work is based on:
P. Luc, C. Couprie, S. Chintala, and J. Verbeek, "Semantic segmentation using adversarial networks," arXiv preprint arXiv:1611.08408, 2016.


INFORMATION:
- The codes have been separated into different files for easy readability
- The Models_Adv.py file contains slightly modified models (UNET, FCN-8 and SegNet) for Adversarial Training as the ground truth needs to be fed in along with the input for loss calculation. This does not change the number of model parameter
