import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import os
from main import load_data, extract_silhouettes, extract_geis, train_model

class MainApplication(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Human Gait Recognition System")
        self.geometry("900x500")
        self.resizable(width=True, height=True)
        self.configure(bg="white")

        # Configure CTk settings for appearance
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # Left panel
        self.left_frame = ctk.CTkFrame(self, width=250, height=600, corner_radius=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        self.left_frame.pack_propagate(False)

        # Right panel
        self.right_frame = ctk.CTkFrame(self, width=700, height=580, corner_radius=10)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        # Title for the left frame
        self.left_panel_title = ctk.CTkLabel(self.left_frame, text="Control Panel", font=("Helvetica", 24, "bold"))
        self.left_panel_title.pack(pady=(20, 10))

        # Buttons in the left frame
        self.btn_file_selection = ctk.CTkButton(self.left_frame, text="File Selection and Processing", command=self.show_tab1, corner_radius=10, width=190, height=35)
        self.btn_file_selection.pack(fill=None, pady=(35, 20), padx=20)

        self.btn_testing = ctk.CTkButton(self.left_frame, text="Gait Prediction", command=self.show_tab2, corner_radius=10, width=190, height=35)
        self.btn_testing.pack(fill=None, pady=5, padx=20)

        # Content for tab 1
        self.tab1_content = ctk.CTkFrame(self.right_frame, corner_radius=10)
        self.create_tab1()

        # Content for tab 2
        self.tab2_content = ctk.CTkFrame(self.right_frame, corner_radius=10)
        self.create_tab2()

        # Initially display tab 1 content
        self.tab1_content.pack(fill=tk.BOTH, expand=True)

    def create_tab1(self):
        # Label and entry for dataset path
        ctk.CTkLabel(self.tab1_content, text="Enter Image Dataset root path (Required*):", font=("Helvetica", 14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dataset_path_entry = ctk.CTkEntry(self.tab1_content, width=250)
        self.dataset_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Browse button for dataset path
        browse_button = ctk.CTkButton(self.tab1_content, text="Browse", command=self.on_browse_dataset, width=70)
        browse_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Load data button
        load_data_button = ctk.CTkButton(self.tab1_content, text="Load Data", command=self.on_load_data)
        load_data_button.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="n")

        # Status label for data loading
        self.status_label = ctk.CTkLabel(self.tab1_content, text="", width=300, anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # Radio buttons for processing steps
        self.processing_var = tk.StringVar()
        self.processing_var.set("Silhouette")

        silhouette_radio = ctk.CTkRadioButton(self.tab1_content, text="Silhouette Extraction", variable=self.processing_var, value="Silhouette", command=self.on_radio_change)
        silhouette_radio.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        gei_radio = ctk.CTkRadioButton(self.tab1_content, text="GEI Extraction", variable=self.processing_var, value="GEI", command=self.on_radio_change)
        gei_radio.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        train_radio = ctk.CTkRadioButton(self.tab1_content, text="Train Model", variable=self.processing_var, value="Train", command=self.on_radio_change)
        train_radio.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        # Start Processing button
        start_processing_button = ctk.CTkButton(self.tab1_content, text="Start Processing", command=self.on_start_processing)
        start_processing_button.grid(row=6, column=1, columnspan=2, padx=5, pady=20, sticky="n")

        # Adjust the entire grid to center in the right frame
        self.tab1_content.grid_rowconfigure(0, weight=0)
        self.tab1_content.grid_rowconfigure(1, weight=0)
        self.tab1_content.grid_rowconfigure(2, weight=0)
        self.tab1_content.grid_rowconfigure(3, weight=0)
        self.tab1_content.grid_rowconfigure(4, weight=0)
        self.tab1_content.grid_rowconfigure(5, weight=0)
        self.tab1_content.grid_rowconfigure(6, weight=1)  # Ensure the remaining space is taken by the last row
        self.tab1_content.grid_columnconfigure(0, weight=1)
        self.tab1_content.grid_columnconfigure(3, weight=1)

    def create_tab2(self):
        ctk.CTkLabel(self.tab2_content, text="Select test images:").pack(pady=10)
        ctk.CTkButton(self.tab2_content, text="Select Images", command=self.on_select_images).pack(pady=10)
        self.test_status_label = ctk.CTkLabel(self.tab2_content, text="", width=300, anchor="w")
        self.test_status_label.pack(pady=20, fill="x")

    def show_tab1(self):
        self.tab2_content.pack_forget()
        self.tab1_content.pack(fill=tk.BOTH, expand=True)

    def show_tab2(self):
        self.tab1_content.pack_forget()
        self.tab2_content.pack(fill=tk.BOTH, expand=True)

    def on_load_data(self):
        path = self.dataset_path_entry.get()
        if not path:
            self.status_label.configure(text="Please enter a dataset path.", text_color="red")
            return
        success, message = load_data(path)
        self.status_label.configure(text=message, text_color="green" if success else "red")

    def on_browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.dataset_path_entry.delete(0, tk.END)
            self.dataset_path_entry.insert(0, path)

    def on_radio_change(self):
        pass

    def on_start_processing(self):
        selected = self.processing_var.get()
        if selected == "Silhouette":
            self.on_extract_silhouettes()
        elif selected == "GEI":
            self.on_extract_gei()
        elif selected == "Train":
            self.on_train_model()

    def on_extract_silhouettes(self):
        success, message = extract_silhouettes()
        self.status_label.configure(text=message, text_color="green" if success else "red")

    def on_extract_gei(self):
        success, message = extract_geis()
        self.status_label.configure(text=message, text_color="green" if success else "red")

    def on_train_model(self):
        success, message = train_model()
        self.status_label.configure(text=message, text_color="green" if success else "red")
    
    def on_select_images(self):
        filetypes = (("Image files", "*.jpg *.png"), ("All files", "*.*"))
        paths = filedialog.askopenfilenames(title="Open images", initialdir="/", filetypes=filetypes)
        if paths:
            self.test_status_label.configure(text=f"Selected images: {paths}")
            # results = test_model(paths)
            # self.show_results(results)

    def show_results(self, results):
        result_window = ctk.CTkToplevel(self)
        result_window.title("Prediction Results")
        result_window.geometry("800x600")
        result_window.configure(bg="white")

        result_frame = ctk.CTkFrame(result_window)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for idx, (image_path, predicted_label, original_label, is_correct) in enumerate(results):
            image_label = ctk.CTkLabel(result_frame, text=f"Image: {os.path.basename(image_path)}")
            image_label.grid(row=idx, column=0, padx=10, pady=5)

            predicted_label = ctk.CTkLabel(result_frame, text=f"Predicted: {predicted_label}")
            predicted_label.grid(row=idx, column=1, padx=10, pady=5)

            original_label = ctk.CTkLabel(result_frame, text=f"Original: {original_label}")
            original_label.grid(row=idx, column=2, padx=10, pady=5)

            result = ctk.CTkLabel(result_frame, text="Correct" if is_correct else "Incorrect", text_color="green" if is_correct else "red")
            result.grid(row=idx, column=3, padx=10, pady=5)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
