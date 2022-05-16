import cv2
import numpy as np
import math
import product
import random
import refining
import char_detection
import total_plate
import total_char
import matplotlib.pyplot as plt

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []
    intCountOfPossibleChars = 0
    imgThreshCopy = imgThresh.copy()
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    for i in range(0, len(contours)):
        
        cv2.drawContours(imgContours, contours, i, product.SCALAR_WHITE)
        possibleChar = total_char.PossibleChar(contours[i])
        if char_detection.checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            listOfPossibleChars.append(possibleChar)  
    
    print("\nstep 2 - len(contours) = " + str(len(contours)))
    print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))
    plt.title("2a")
    plt.imshow(imgContours)
    plt.show()
    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = total_plate.PossiblePlate()
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    intTotalOfCharHeights = 0
    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = char_detection.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    height, width, numChannels = imgOriginal.shape
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    possiblePlate.imgPlate = imgCropped
    return possiblePlate

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []
    height, width, numChannels = imgOriginalScene.shape
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
    cv2.destroyAllWindows()
    plt.title("0")
    plt.imshow(imgOriginalScene)
    plt.show()
    imgGrayscaleScene, imgThreshScene = refining.preprocess(imgOriginalScene)
    plt.title("1a")
    plt.imshow(imgGrayscaleScene)
    plt.show()
    
    plt.title("1b")
    plt.imshow( imgThreshScene)
    plt.show()
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))
    imgContours = np.zeros((height, width, 3), np.uint8)
    contours = []
    for possibleChar in listOfPossibleCharsInScene:
        contours.append(possibleChar.contour)
    cv2.drawContours(imgContours, contours, -1, product.SCALAR_WHITE)
    plt.title("2b")
    plt.imshow( imgContours)
    plt.show()
    listOfListsOfMatchingCharsInScene = char_detection.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    

    print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene)))
    imgContours = np.zeros((height, width, 3), np.uint8)
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        intRandomBlue = random.randint(0, 255)
        intRandomGreen = random.randint(0, 255)
        intRandomRed = random.randint(0, 255)
        contours = []
        for matchingChar in listOfMatchingChars:
            contours.append(matchingChar.contour)
        cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
    plt.title("3")
    plt.imshow(imgContours)
    plt.show()
        
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)
        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)
    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  
    
    print("\n")
    plt.title("4a")
    plt.imshow(imgContours)
    plt.show()
    
    for i in range(0,len(listOfPossiblePlates)):        
        print("possible plate " + str(i))
        plt.title("4b")
        plt.imshow( listOfPossiblePlates[i].imgPlate)
        plt.show()
    
    print("\nplate detection complete....\n")

    return listOfPossiblePlates
