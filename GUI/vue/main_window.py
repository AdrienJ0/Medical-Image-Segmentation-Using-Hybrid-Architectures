from GUI.controller.utils import *
from GUI.vue.home.select_model import ModelSelect
from GUI.vue.home.open_file import OpenFile


class MainWindow(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(height=master.app_height, width=master.app_width, fg_color=("gray10", "gray10"))
        self.model = None
        self.model_select = ModelSelect(master=self)
        self.model_select.place(x=20, relx=0, rely=0.5, anchor="w")
        self.open_file = OpenFile(master=self)
        self.open_file.place(x=-20, relx=1, rely=0.5, anchor="e")

    def switch_state(self, path):
        self.master.pack_image_display(self.model, path)
