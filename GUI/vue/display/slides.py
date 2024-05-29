from GUI.controller.utils import *


class SlidesView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=("gray10", "gray10"), height=500, width=800)
        self.raw_label = ctk.CTkLabel(master, text="Raw image", font=("Arial", 20, "bold"))
        self.raw_label.place(relx=0.2, rely=0.1, anchor="w")
        self.mask_label = ctk.CTkLabel(master, text="Mask image", font=("Arial", 20, "bold"))
        self.mask_label.place(relx=0.8, rely=0.1, anchor="e")

        self.slides, self.masks = self.load_slides(master.path)

        self.current_slide = tk.IntVar(value=0)

        self.raw_slide = self.slides[0]
        self.image_label = ctk.CTkLabel(master, image=self.raw_slide, text="")
        self.image_label.place(relx=0.1, rely=0.5, anchor="w")

        self.mask_slide = self.masks[0]
        self.mask_label = ctk.CTkLabel(master, image=self.mask_slide, text="")
        self.mask_label.place(relx=0.9, rely=0.5, anchor="e")

        self.slider = ctk.CTkSlider(master=self, variable=self.current_slide, from_=0, to=len(self.slides)-1, number_of_steps=len(self.slides)-1, command=self.update_slide)
        self.slider.place(relx=0.5, rely=0, anchor="n", y=20)

    def update_slide(self, value):
        self.raw_slide = self.slides[int(value)]
        self.image_label.configure(image=self.raw_slide)

        self.mask_slide = self.masks[int(value)]
        self.mask_label.configure(image=self.mask_slide)

    def load_slides(self, path):
        path_list = path.split("/")
        if path_list[-1].split(".")[-1] == "npz":
            # get all the slices associated with the npz file
            case = path_list[-1].split("_")[0]
            slices = [f for f in os.listdir("\\".join(path_list[0:-1])) if case in f]
            masks = list(map(lambda x: "\\".join(path_list[0:-1]) + "\\" + x, slices))
            # transform the slices to ctk images
            slices = list(map(lambda x: npz_to_image(x), masks))
            return slices, slices

        elif path_list[-1].split(".")[-1] == "png":
            slices = [f.split(".")[0] for f in os.listdir("\\".join(path_list[0:-1])) if f.split(".")[0].isdigit()]
            slices.sort()
            # to adapt for each nomenclature of computed masks
            paths = list(map(lambda x: "\\".join(path_list[0:-1]) + "\\" + x + ".png", slices))
            masks = list(map(lambda x: "\\".join(path_list[0:-1]) + "\\" + x + "_mask2.png", slices))
            masks = list(map(lambda x: ctk.CTkImage(Image.open(x), size=(512, 512)), masks))
            slices = list(map(lambda x: ctk.CTkImage(Image.open(x), size=(512, 512)), paths))
            return slices, masks




