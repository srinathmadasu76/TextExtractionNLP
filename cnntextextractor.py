# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:42:52 2020

@author: Srinath
"""


import re

from pkg_resources import resource_filename
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
import pytest
from pytextractor.pytextractor import PyTextractor
extractor = PyTextractor()
detected = extractor.get_image_text('C:/Users/Srinath.000/Desktop/Srinath_CV/hackerrank/j4.png', width=32,min_boxes=2)
print(detected)
# class TestDetector(object):
#     def setup_method(self, method):
#         self.extractor = PyTextractor()

    # @pytest.mark.parametrize('image', [
    #     resource_filename(__name__, 'test_images/52.png'),
    #     resource_filename(__name__, 'test_images/53.png'),
    #     resource_filename(__name__, 'test_images/56.png'),
    # ])
    # def test_number_detector(self, image):
    #     detected = self.extractor.get_image_text(image, number=True)
    #     assert len(detected)

    # @pytest.mark.parametrize('image,expected_number', [
    #     (resource_filename(__name__, 'test_images/52.png'), 52),
    # ])
    # def test_number_extractor(self, image, expected_number):
    #     detected = self.extractor.get_image_text(image, number=True)
    #     assert(int(detected[0]) == expected_number)


    # @pytest.mark.skip(reason="[LIMITATION] can't detect number")
    # @pytest.mark.parametrize('image,expected_number', [
    #     (resource_filename(__name__, 'test_images/53.png'), 53),
    #     (resource_filename(__name__, 'test_images/56.png'), 56),
    #     (resource_filename(__name__, 'test_images/57.png'), 57),
    # ])
    # def test_number_extractor_tweaked(self, image, expected_number):
    #     detected = self.extractor.get_image_text(image, number=True, percentage=4, min_boxes=2)
    #     for roi in detected:
    #         try:
    #             print(roi, ' ? ', expected_number)
    #             number = int(re.sub(r'\D', '', roi))
    #             if number == expected_number:
    #                 break
    #         except ValueError:
    #             pass
    #     else:
    #         assert False
    