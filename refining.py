import cv2
import numpy as np
import math
import matplotlib.pyplot as plt



GUASSIAN_SMOOTH_FILTER=(5,5)
ADAPTIVE_THRES_BLOCK_SIZE=19
ADAPTIVE_THRES_WEIGTH=9


def maximizeContrast(imgGrayScale):
    height,width=imgGrayScale.shape
    imgTopHat=np.zeros((height,width,1),np.uint8)
    imgBlackHat=np.zeros((height,width,1),np.uint8)
    structuringElement=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    imgTopHat=cv2.morphologyEx(imgGrayScale,cv2.MORPH_TOPHAT,structuringElement)
    imgBlackHat=cv2.morphologyEx(imgGrayScale,cv2.MORPH_BLACKHAT,structuringElement)
    imgGrayScaleplusTopHat=cv2.add(imgGrayScale,imgTopHat)
    imgGrayScaleplusTopHatminusBlackHat=cv2.subtract(imgGrayScaleplusTopHat,imgBlackHat)
    return imgGrayScaleplusTopHatminusBlackHat


def preprocess(imgOriginal):
    imgGrayScale=extractValue(imgOriginal)
    imgMaxContrastGrayScale=maximizeContrast(imgGrayScale)
    height,width=imgGrayScale.shape
    imgBlurred=np.zeros((height,width,1),np.uint8)
    imgBlurred=cv2.GaussianBlur(imgMaxContrastGrayScale,GUASSIAN_SMOOTH_FILTER,0)
    imgThresh=cv2.adaptiveThreshold(imgBlurred,255.0,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,ADAPTIVE_THRES_BLOCK_SIZE,ADAPTIVE_THRES_WEIGTH)
    return imgGrayScale,imgThresh


def extractValue(imgOriginal):
    height, width,numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue


    =]
    