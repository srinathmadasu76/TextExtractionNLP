# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:29:34 2020

@author: Srinath
"""


# adds image processing capabilities
#from PIL import Image
# will convert the image to text string
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
# assigning an image from the source path
#img = Image.open('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/1114c.png')
# converts the image to result and saves it into result variable
#result = pytesseract.image_to_string(img)
#with open('text_result.txt',mode='w') as file:
 #file.write(result)
 #print("ready")
 # adds more image processing capabilities
from PIL import Image, ImageEnhance
# preprocessing
# gray scale
#def pytesseractresult(img1):
def gray(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png",img)
    return img

# blur
def blur(img) :
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite(r"./preprocess/img_blur.png",img)    
    return img_blur

# threshold
def threshold(img):
    #pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
    cv2.imwrite(r"./preprocess/img_threshold.png",img)
    return img
# assigning an image from the source path
img = Image.open('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/1114c.png')
img_cv = cv2.imread('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/111b.png')
#img_cv = cv2.imread(img1)
# Finding contours 
im_gray = gray(img_cv)
im_blur = blur(im_gray)
im_thresh = threshold(im_blur)

contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(im_thresh, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        #cv2.imshow('cnt',rect)
        #cv2.waitKey()

        # Cropping the text block for giving input to OCR 
        cropped = im_thresh[y:y + h, x:x + w] 
# adding some sharpness and contrast to the image 
img_cv = cv2.imread('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/111b.png')
enhancer1 = ImageEnhance.Sharpness(img)
enhancer2 = ImageEnhance.Contrast(img)
img_edit = enhancer1.enhance(40.0)
img_edit = enhancer2.enhance(2.5)
# save the new image
img_edit.save("edited_image.png")
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# converts the image to result and saves it into result variable
custom_oem_psm_config = r'--psm 6'
result = pytesseract.image_to_string(img_rgb,lang='eng', config=custom_oem_psm_config)
result = pytesseract.image_to_string(cropped,lang='eng', config=custom_oem_psm_config)

with open('text_result2.jpg',mode='w', encoding="utf-8") as file:
 file.write(result)
 print("ready")
     
     #return result
     
