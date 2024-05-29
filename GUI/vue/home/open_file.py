from GUI.controller.utils import *


class OpenFile(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=("gray50", "gray40"), height=FULL_HEIGHT*0.9, width=FULL_WIDTH*0.3)
        file_img = ctk.CTkImage(Image.open("../GUI/icons/export.png"), size=(100, 100))
        self.open_file_button = ctk.CTkButton(master=self, text="Choose a file to visualize", command=self.load_image_file, image=file_img, compound="top")
        self.open_file_button.place(relx=0.5, rely=0.5, anchor="center")

    def load_image_file(self):
        path = browse_files(self.master.model)
        if path:
            self.master.switch_state(path)
