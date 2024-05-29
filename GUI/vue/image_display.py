from GUI.controller.utils import *
from GUI.vue.display.slides import SlidesView
from GUI.vue.display.go_back import GoBack


class ImageDisplay(ctk.CTkFrame):
    def __init__(self, master, model, path, **kwargs):
        super().__init__(master, **kwargs)
        self.model = model
        self.path = path
        self.configure(height=master.app_height, width=master.app_width, fg_color=("gray10", "gray10"))

        self.slides = SlidesView(master=self)
        self.slides.place(relx=0.5, rely=0.5, anchor="center")

        self.go_back = GoBack(master=self)
        self.go_back.place(relx=1, rely=0, anchor="e", x=-20, y=20)

