import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os

FULL_HEIGHT = 780
FULL_WIDTH = 1620

model_to_dir = {
    "Your model here": "your path here", # "/Predictions_Exam/Predictions_Exam/ParaTransCNN_CA_SA",
}


def browse_files(model):
    """Browse files function"""
    print(model_to_dir.get(model, "./data/Synapse"))
    filename = filedialog.askopenfilename(initialdir=model_to_dir.get(model, "./data/Synapse"),
                                          title="Select a File",
                                          filetypes=(("npz files", "*.npz"),
                                                     ("png files", "*.png"),
                                                     ("numpy files", "*.npy*"),
                                                     ("all files", "*.*")))
    return filename


def npz_to_image(path):
    """Read npz file function"""
    with np.load(path) as data:
        image = data['image']

    return ctk.CTkImage(Image.fromarray(image*255), size=(512, 512))


def read_nii(path):
    return
