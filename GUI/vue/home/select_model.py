from GUI.controller.utils import *


class ModelButton(ctk.CTkRadioButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="gray50", corner_radius=20, border_width_checked=10)


class ModelSelect(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(height=FULL_HEIGHT*0.9, width=FULL_WIDTH*0.6, fg_color=("gray50", "gray40"))

        computed_ph = ctk.CTkFrame(master=self, fg_color=("gray20", "gray20"), height=FULL_HEIGHT*0.85, width=FULL_WIDTH*0.25)
        computed_label = ctk.CTkLabel(master=computed_ph, text="Computed results", font=("Arial", 25, "bold"))
        computed_label.place(relx=0.5, rely=0.1, anchor="center")
        computed_ph.place(relx=0, rely=0.5, anchor="w", x=10)

        to_compute_ph = ctk.CTkFrame(master=self, fg_color=("gray20", "gray20"), height=FULL_HEIGHT*0.85, width=FULL_WIDTH*0.25)
        to_compute_label = ctk.CTkLabel(master=to_compute_ph, text="Masks to compute", font=("Arial", 25, "bold"))
        to_compute_label.place(relx=0.5, rely=0.1, anchor="center")
        to_compute_ph.place(relx=1, rely=0.5, anchor="e", x=-10)

        self.radio_var = tk.StringVar(value="TransUnet")
        model_1 = ModelButton(computed_ph, text="ParaTransCNN_CA_SA", command=self.select_model, variable=self.radio_var, value="ParaTransCNN_CA_SA")
        model_2 = ModelButton(computed_ph, text="ParaTransCNN_SA", command=self.select_model, variable=self.radio_var, value="ParaTransCNN_SA")
        model_3 = ModelButton(computed_ph, text="ParaTransCNN5", command=self.select_model, variable=self.radio_var, value="ParaTransCNN5")
        model_4 = ModelButton(to_compute_ph, text="Trans_UNet", command=self.select_model, variable=self.radio_var, value="Trans_UNet")
        model_5 = ModelButton(to_compute_ph, text="UNet", command=self.select_model, variable=self.radio_var, value="UNet")
        model_6 = ModelButton(to_compute_ph, text="Trans_Unet_perso", command=self.select_model, variable=self.radio_var, value="Trans_Unet_perso")

        model_1.place(relx=0.2, rely=0.3, anchor="w")
        model_2.place(relx=0.2, rely=0.5, anchor="w")
        model_3.place(relx=0.2, rely=0.7, anchor="w")
        model_4.place(relx=0.2, rely=0.3, anchor="w")
        model_5.place(relx=0.2, rely=0.5, anchor="w")
        model_6.place(relx=0.2, rely=0.7, anchor="w")

    def select_model(self):
        self.master.model = self.radio_var.get()

