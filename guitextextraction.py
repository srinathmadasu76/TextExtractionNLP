# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:04:39 2020

@author: Srinath
"""


from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image
import pytesseract
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

from handwriting_recognition import pytesseractresult


class PyTextractor:
    def __init__(self, master):
        self.master = master
        master.title("Text Extraction")
        master.geometry("500x500")
        master.resizable(0, 0)
        root.configure(bg='blue4')

        self.select_button = Button(master, text="Select Method")
        self.select_button.pack()
        self.cmb = ttk.Combobox(master,width="15",values=("PyTesseract", "PyTextractor"))
        self.cmb.pack()
        self.textify_button = Button(master, text="Open Image", command=self.textify)
        self.textify_button.pack()
    def textify(self):
        target = filedialog.askopenfilename()
        if self.cmb.get()=="PyTextractor":
            textinimage = image_to_string(Image.open(target))
        else:
            textinimage = pytesseractresult((target))
        T.insert(END, textinimage)

root = Tk()
#root.configure(bg='blue4')
S = Scrollbar(root)
T = Text(root, height=4, width=50)
S.pack(side=RIGHT, fill=Y)
T.pack(side=LEFT, fill=Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)

graphical = PyTextractor(root)
def quit():
   root.destroy()

btn1=ttk.Button(root, text="Exit", command=quit)
btn1.place(relx="0.2",rely="0.8")
root.mainloop()