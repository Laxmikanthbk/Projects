import sys
import pyperclip
import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
import product



try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True




def number_plate_out(image):
    product.main(image)
    image = cv2.imread(image)

    image = imutils.resize(image, width=500)

    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    edged = cv2.Canny(gray, 170, 200)
    

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
    NumberPlateCnt = None 

    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

    
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
    new_image = cv2.bitwise_and(image,image,mask=mask)
    cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    


    config = ('-l eng --oem 1 --psm 3')


    text = pytesseract.image_to_string(new_image, config=config)
    print("characters detected :",text)
    
    df=pd.read_csv("data.csv")
    a=df[df['ID']==text]
    if a.empty:
        second=tk.Tk()
        second.title("Result")
        second.geometry("300x300")
        tk.Label(second,text="Number plate data not found").grid(row=2,column=1)
        
        
    else:
        a=a.values.tolist()
        a=a[0]
    
        second=tk.Tk()
        second.title("Result")
        second.geometry("300x300")
        tk.Label(second,text="Number is: %s "%text).grid(row=1,column=1)
        tk.Label(second,text="Owner is: %s"%a[1]).grid(row=2,column=1)
        tk.Label(second,text="Model is: %s"%a[2]).grid(row=3,column=1)
        tk.Label(second,text="Insurance status: %s"%a[3]).grid(row=4,column=1)
        tk.Label(second,text="Fines Due Rupees: %s"%a[4]).grid(row=5,column=1)
        
    
    return
















def BROWSE():
    print('test_support.BROWSE')
    sys.stdout.flush()
    
def ONLY_PLATE():
    print('test_support.ONLY_PLATE')
    sys.stdout.flush()
    
    
def SHOW_FULL_STEPS():
    print('test_support.SHOW_FULL_STEPS')
    sys.stdout.flush()
    
    
def START():
    img=pyperclip.paste()
    number_plate_out(img)
    
def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top
    
def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


