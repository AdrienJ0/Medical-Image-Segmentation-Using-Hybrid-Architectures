from GUI.controller.utils import *


class GoBack(ctk.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=("gray10", "gray10"), height=20, width=50)

        self.go_back_button = ctk.CTkButton(master=self, text="Go back", command=self.go_back)
        self.go_back_button.place(relx=0.5, rely=0.5, anchor="center")

    def go_back(self):
        self.master.master.pack_home()




