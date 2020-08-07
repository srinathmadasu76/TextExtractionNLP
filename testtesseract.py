# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:25:38 2020

@author: Srinath
"""


import pytesseract
from PIL import Image
img = Image.open('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/virginia-quote.jpeg')
# converts the image to result and saves it into result variable
result = pytesseract.image_to_string(img)