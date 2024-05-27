import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk 

def load_data(path):
    print(f"Loading data from {path}")
    return True

def load_saved_data():
    print("Loading saved data from pickle")
    return True

def test_model(image_paths):
    print(f"Testing model on {image_paths}")
    return True

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
        self.left_panel_title = ctk.CTkLabel(self.left_frame, text="Control Panel", font=("Helvetica", 24, "bold"))  # Adjust the font and colors as needed
        self.left_panel_title.pack(pady=(20, 10))  # Padding for top and bottom spacing

        # Buttons in the left frame
        self.btn_file_selection = ctk.CTkButton(self.left_frame, text="File Selection and Processing", command=self.show_tab1, corner_radius=10, width=190, height = 35)
        self.btn_file_selection.pack(fill=None, pady=(35, 20), padx=20)  # Adjust vertical padding to position below the title

        self.btn_testing = ctk.CTkButton(self.left_frame, text="Testing", command=self.show_tab2, corner_radius=10, width=190, height = 35)
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
        ctk.CTkLabel(self.tab1_content, text="Enter dataset root path:", font=("Helvetica", 14)).grid(row=1, column=0, padx=10, pady=(20, 10), sticky="e")
        self.dataset_path_entry = ctk.CTkEntry(self.tab1_content, width=200)
        self.dataset_path_entry.grid(row=1, column=1, padx=10, pady=(20, 10), sticky="w")

        # Buttons for loading data
        load_data_button = ctk.CTkButton(self.tab1_content, text="Load Data", command=self.on_load_data)
        load_data_button.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        load_saved_data_button = ctk.CTkButton(self.tab1_content, text="Load Saved Data", command=self.on_load_saved_data)
        load_saved_data_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Status text box
        self.status_text = ctk.CTkTextbox(self.tab1_content, height=10, width=300)
        self.status_text.grid(row=3, column=0, columnspan=2, padx=10, pady=20)

        # Adjust the entire grid to center in the right frame
        self.tab1_content.grid_rowconfigure(0, weight=1)  # Empty space above widgets
        self.tab1_content.grid_rowconfigure(4, weight=1)  # Empty space below widgets
        self.tab1_content.grid_columnconfigure(0, weight=1)
        self.tab1_content.grid_columnconfigure(2, weight=1)


    def create_tab2(self):
        ctk.CTkLabel(self.tab2_content, text="Select test images:").pack(pady=10)
        ctk.CTkButton(self.tab2_content, text="Select Images", command=self.on_select_images).pack(pady=10)
        self.test_status_text = ctk.CTkTextbox(self.tab2_content, height=10, width=300)
        self.test_status_text.pack(pady=20)

    def show_tab1(self):
        self.tab2_content.pack_forget()
        self.tab1_content.pack(fill=tk.BOTH, expand=True)

    def show_tab2(self):
        self.tab1_content.pack_forget()
        self.tab2_content.pack(fill=tk.BOTH, expand=True)

    def on_load_data(self):
        path = self.dataset_path_entry.get()
        if not path:
            ctk.CTkMessagebox.show_error("Error", "Please enter a dataset path.")
            return
        if load_data(path):
            self.status_text.insert(tk.END, "Data loaded successfully.\n")
        else:
            self.status_text.insert(tk.END, "Failed to load data.\n")

    def on_load_saved_data(self):
        if load_saved_data():
            self.status_text.insert(tk.END, "Saved data loaded successfully.\n")
        else:
            self.status_text.insert(tk.END, "Failed to load saved data.\n")

    def on_select_images(self):
        filetypes = (("Image files", "*.jpg *.png"), ("All files", "*.*"))
        paths = filedialog.ask


    def on_select_images(self):
        filetypes = (("Image files", "*.jpg *.png"), ("All files", "*.*"))
        paths = filedialog.askopenfilenames(title="Open images", initialdir="/", filetypes=filetypes)
        if paths:
            self.test_status_text.insert(tk.END, f"Selected images: {paths}\n")
            if test_model(paths):
                self.test_status_text.insert(tk.END, "Model test completed successfully.\n")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()