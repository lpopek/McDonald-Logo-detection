# McDonald Logo detection

![Result of Detection](documentation\final_result.png )

## Introduction

App created for automatic McDonald logo detection, based on traditional semantic aproach. It is final solution for WUT course of Digital Image Processing (POBR) for Winter semester 2021. The application contains a stand-alone implementation of the algorithms used.

## Build with

* [Python](https://docs.python.org/3/)
* [OpenCV](https://opencv.org/) for  and writing images
* [Matplotlib](https://matplotlib.org/) for visualising effect of detection.
* [NumPY](https://numpy.org/doc/stable/index.html) for matrix operation

## Prerequsites

This is an example of how to list things you need to use the software and how to install them.

* OpenCV version 4.5.1

  ```sh
  pip install cv2
  ```

* Matplotlib version 3.4.2

  ```sh
  pip install matplotlib
  ```

* Numpy version 1.20.3

  ```sh
  pip install numpy
  ```

## Installation and usage

Tested for Windows 10

1. Clone repo

   ```sh
   git clone https://github.com/lpopek/McDonald-Logo-detection.git
   ```

2. In clone directory write command

   ```sh
   detect_mcdonald_logo.py -i <image_path> -s
   ```

3. Check detection results in .\save directory

## Algorithm description in steps

1. Conversion image from RGB to HSV color space. (preprocessing module)

2. Tresholding of Hue range typical for McDonald logo. (preprocessing module)

3. Closing operation (erosion and after dilation) on image after tresholding. (preprocessing module)

4. Segmentation using Floodfill algorithm. (segmentation module)

5. Countour tracing using Moore-Neighbourhood algorithm. (segmentation module)

6. Feature vectors calculation using geometrical invariants and moments. It enables to distinguish the same objects even if they are in different scale or rotation. (classification module)

7. Classification using modified K- nearest neighbours algorithm. In this case is classification proper logo vs other. Config.json file contains results from caculating average vector from dataset collected(classification module)

## Command line flags

* **-i** - input image
* **-n** - take image from dataset uploaded

Optional:

* **-s** - save result
* **-d** - display all steps during segmentation and detection proces
* **-h** - display help
