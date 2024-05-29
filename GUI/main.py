import sys

import customtkinter as ctk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import SimpleITK as sitk
import numpy as np
from vue.main_window import MainWindow
from vue.image_display import ImageDisplay


class App(ctk.CTk):
    """Main application class."""
    def __init__(self):
        super().__init__()
        ctk.deactivate_automatic_dpi_awareness()  # bug source of not well scaling
        self.app_width = self.winfo_screenwidth() - 300
        self.app_height = self.winfo_screenheight() - 300
        self.geometry(f"{self.app_width}x{self.app_height}")  # min window size
        self.minsize(400, 400)
        self.main_w = None
        self.image_display = None
        self.title("Model Viewer")

        # initiate main app
        self.update_idletasks()

        # initiate main app
        self.pack_home()

    def pack_home(self):
        for widgets in self.winfo_children():
            widgets.destroy()
        self.main_w = MainWindow(master=self)
        self.main_w.pack(expand=True, fill="both")

    def pack_image_display(self, model, path):
        for widgets in self.winfo_children():
            widgets.destroy()
        self.image_display = ImageDisplay(master=self, model=model, path=path)
        self.image_display.pack(expand=True, fill="both")


if __name__ == "__main__":
    """main entry point"""
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")

    app = App()
    app.state("zoomed")
    app.mainloop()
