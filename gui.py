import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk, ImageOps
import numpy as np
import cv2
import threading, asyncio
import ImageRecognition
from pathlib import Path
from ImageRecognition.image_recognizer import ImageRecognizer, ModelType
from ImageRecognition.segmentation import (segment_contours, segment_connected, segment_projection)
from ImageRecognition.segmentation import (segment_contours, segment_connected, segment_projection, segment_watershed)
from ImageRecognition.models import ModelBase
import torch
from torchvision.transforms import ToTensor, Compose, Normalize

'''MODEL_PATH = Path(__file__, "ImageRecognition", "model", "model.pth")'''
MODEL_DIR = Path(__file__).parent / "ImageRecognition" / "model"

class DigitDrawingApp:
    def __init__(self, root, model_dir = None):
        self.root = root
        self.root.title("Digit Drawing and Preprocessing Demo")

        # Canvas size
        self.canvas_width = 280
        self.canvas_height = 280

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white', cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Buttons
        self.clear_btn = ttk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=1, column=0, pady=10)

        self.process_btn = ttk.Button(root, text="Process", command=self.process_drawing)
        self.process_btn.grid(row=1, column=1, pady=10)


        self.predict_btn = ttk.Button(root, text="Predict", command=self.predict_drawing)
        self.predict_btn.grid(row=1, column=2, pady=10)

        self.predict_result = tk.StringVar(root, value="Prediction: None")
        self.predict_result_label = tk.Label(root, textvariable=self.predict_result)
        self.predict_result_label.grid(row=3, column=2, pady=10)

        self.train_btn = ttk.Button(root, text="Train", command=self.train)
        self.train_btn.grid(row=1,column=3, pady=10)

        self.epochs = tk.IntVar(root, value=5)

        self.train_entry = ttk.Entry(root, textvariable=self.epochs)
        self.train_entry.grid(row=1,column=4, pady=10)


        # Panel for processed images (Segmented images included)
        self.processed_frame = tk.Frame(root)
        self.processed_frame.grid(row=2, column=0, columnspan=5, pady=10)

        # Labels for showing processed images
        self.gray_label = tk.Label(self.processed_frame, text="Gray")
        self.gray_label.grid(row=0, column=0, padx=5, pady=5)

        self.binary_label = tk.Label(self.processed_frame, text="Binary")
        self.binary_label.grid(row=0, column=1, padx=5, pady=5)

        self.edge_label = tk.Label(self.processed_frame, text="Edges")
        self.edge_label.grid(row=0, column=2, padx=5, pady=5)


        self.segmentation_options = ["","Projection", "Contour", "Connected","Watershed (Aggressive)"]
        self.segmentation_stringvar = tk.StringVar(value="Projection")
        self.segmentation_menu = ttk.OptionMenu(root,self.segmentation_stringvar, *self.segmentation_options)
        self.segmentation_menu.grid(row=4,column=5)


        self.model_options = ["", "CNN", "MLP", "SVC"]
        print(*self.model_options)
        self.model_stringvar = tk.StringVar(value="CNN")
        self.model_menu = ttk.OptionMenu(root, self.model_stringvar, *self.model_options)
        self.model_menu.grid(row=5, column=5)
        
        



        # Labels for showing the segmented images
        """self.contour_label = tk.Label(self.processed_frame, text="Contours")
        self.contour_label.grid(row=0, column=3, padx=5, pady=5)

        self.connected_label = tk.Label(self.processed_frame, text="Connected")
        self.connected_label.grid(row=0, column=4, padx=5, pady=5)

        self.projection_label = tk.Label(self.processed_frame, text="Projection")
        self.projection_label.grid(row=0, column=5, padx=5, pady=5)"""

        # For drawing
        self.old_x = None
        self.old_y = None
        self.pen_width = 8  # thickness of the drawing pen
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Create a PIL image to draw on
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), 'white')
        self.draw_image = ImageDraw.Draw(self.image)

        
        #self.image_rec = ImageRecognizer(64, model_path) 
        #commented this out to use MLP model ^
        self.image_rec = ImageRecognizer(batch_size=64, model_dir = model_dir)

    def train(self):
        epochs = self.epochs.get() if self.epochs.get() > 0 else 1
        
        def train_sequence(self,model,epochs):
            model.train()
            self.image_rec.save_model()
        threads: list[threading.Thread] = []
        for model_type in [ModelType.CNN,ModelType.MLP,ModelType.SVC]:
            training_thread = threading.Thread(target=train_sequence, args=(self,self.image_rec.models[model_type],epochs))
            threads.append(training_thread)
        for t in threads:
            t.start()

    def draw(self, event):
        if self.old_x and self.old_y:
            # Draw line on both canvas and PIL image
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.pen_width, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw_image.line([self.old_x, self.old_y, event.x, event.y], fill='black', width=self.pen_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), 'white')
        self.draw_image = ImageDraw.Draw(self.image)
        # Clear processed images
        self.gray_label.config(image='')
        self.binary_label.config(image='')
        self.edge_label.config(image='')

    def process_drawing(self):
        # Convert PIL image to grayscale
        gray = ImageOps.grayscale(self.image)
        gray_tk = ImageTk.PhotoImage(gray.resize((100, 100)))
        self.gray_label.config(image=gray_tk)
        self.gray_label.image = gray_tk  # Keep reference

        # Convert to numpy array for further processing
        img_np = np.array(gray)
        
        # Negative
        #img_np = img_np.max() - img_np

        # Normalise
        normalised_img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Binarization
        _, binary = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)
        binary_img = Image.fromarray(binary)
        binary_tk = ImageTk.PhotoImage(binary_img.resize((100, 100)))
        self.binary_label.config(image=binary_tk)
        self.binary_label.image = binary_tk

        # Edge Detection (Canny)
        edges = cv2.Canny(img_np, 100, 200)
        edges_img = Image.fromarray(edges)
        edges_tk = ImageTk.PhotoImage(edges_img.resize((100, 100)))
        self.edge_label.config(image=edges_tk)
        self.edge_label.image = edges_tk

        # --- Segmentation techniques ---
        contour_digits = segment_contours(img_np)
        connected_digits = segment_connected(img_np)
        projection_digits = segment_projection(img_np)
        watershed_digits = segment_watershed(img_np)

        for digits in [contour_digits, connected_digits, projection_digits]:
            for i,digit in enumerate(digits):
                #digits[i] = (digit - digit.min()) / (digit.max() - digit.min())
                pass

        print(f"""
Contour: {len(contour_digits)}
Connected: {len(connected_digits)}
Projection: {len(projection_digits)}""")
Projection: {len(projection_digits)})
Watershed: {len(watershed_digits)}""")

        # Helper to display multiple segmented digits beneath a label
        def display_segmented_digits(digit_list, parent_frame, row, label_text):
            # Clear previous widgets in that row except the first column (label)
            for widget in parent_frame.grid_slaves(row=row):
                widget.grid_forget()

            # Place label at column 0
            label = tk.Label(parent_frame, text=label_text)
            label.grid(row=row, column=0, padx=5, pady=5)

            # Place images starting at column 1
            for i, digit_arr in enumerate(digit_list):
                digit_img = Image.fromarray(digit_arr)
                digit_tk = ImageTk.PhotoImage(digit_img.resize((28, 28)))
                lbl = tk.Label(parent_frame, image=digit_tk)
                lbl.image = digit_tk  # keep reference
                lbl.grid(row=row, column=i+1, padx=2, pady=2)

        # Display all segmented digits in rows beneath the main previews
        display_segmented_digits(contour_digits, self.processed_frame, row=1, label_text="Contours")
        display_segmented_digits(connected_digits, self.processed_frame, row=2, label_text="Connected")
        display_segmented_digits(projection_digits, self.processed_frame, row=3, label_text="Projection")
        display_segmented_digits(watershed_digits, self.processed_frame, row=4, label_text="Watershed")


        """if contour_digits:
            contour_img = Image.fromarray(contour_digits[0])
            contour_tk = ImageTk.PhotoImage(contour_img.resize((100,100)))
            self.contour_label.config(image=contour_tk)
            self.contour_label.image = contour_tk

        if connected_digits:
            connected_img = Image.fromarray(connected_digits[0])
            connected_tk = ImageTk.PhotoImage(connected_img.resize((100,100)))
            self.connected_label.config(image=connected_tk)
            self.connected_label.image = connected_tk

        if projection_digits:
            projection_img = Image.fromarray(projection_digits[0])
            projection_tk = ImageTk.PhotoImage(projection_img.resize((100,100)))
            self.projection_label.config(image=projection_tk)
            self.projection_label.image = projection_tk"""

        return {
            "grey": gray,
            "img": img_np,
            "normalised": normalised_img_np,
            "edges": edges,
            "binary": binary,
            "Connected": connected_digits,
            "Projection": projection_digits,
            "Contour": contour_digits,
            "Watershed (Aggressive)": watershed_digits
        }

    @property
    def active_model(self) -> ModelType:
        value = self.model_stringvar.get()
        match value:
            case "CNN":
                return ModelType.CNN
            case "MLP":
                return ModelType.MLP
            case "SVC":
                return ModelType.SVC
            case _:
                return ModelType.CNN



    def predict_drawing(self):
        print(f"Predicting with model: {self.active_model}")
        
        # This function runs in a separate thread
        def prediction_sequence(self: DigitDrawingApp):
            drawing = self.process_drawing()
            
            # This now contains correctly centered 28x28 numpy arrays
            segmented_digits = drawing[self.segmentation_stringvar.get()]
            preds = ""
            
            # Define the same transformations used for MNIST training data
            # This is the correct transform pipeline from models.py
            mnist_transform = Compose([
                ToTensor(),  # Converts PIL Image [0, 255] to Tensor [0.0, 1.0] and adds Channel dim
                Normalize((0.1307,), (0.3081,)) # MNIST-specific normalization
            ])

            model = self.image_rec.models[self.active_model]
            
            for n in segmented_digits:
                # 'n' is a 28x28 numpy array (white digit on black background)

                if self.active_model != ModelType.SVC:
                    # --- Start Fix ---
                    
                    # 1. Convert numpy array to PIL Image (which ToTensor() expects)
                    # 'L' mode is for 8-bit grayscale
                    pil_image = Image.fromarray(n, mode='L')

                    # 2. Apply the *exact* transform used in training
                    # This scales to [0,1], normalizes, and sets shape to [1, 28, 28] (C, H, W)
                    n_tensor = mnist_transform(pil_image)

                    # 3. Add the batch dimension (B, C, H, W)
                    # Models expect a batch, so we unsqueeze at dim 0
                    n_tensor = n_tensor.unsqueeze(0).to(model.device)
                    # --- End Fix ---
                    
                else: # SVC model
                    # SVC expects the raw, flattened pixel values (0-255)
                    n_tensor = n.reshape(1, -1) # This was already correct

                # Pass the prepared tensor (or array for SVC) to the model
                prediction = self.image_rec.predict(self.active_model, n_tensor)
                preds += str(prediction) + "  "
                self.predict_result.set(f"Prediction: {preds}")

        prediction_thread = threading.Thread(target=prediction_sequence, args=[self])
        prediction_thread.start()
        

def start_app() -> tuple[DigitDrawingApp, threading.Thread]:
    root = tk.Tk()
    app = DigitDrawingApp(root, MODEL_DIR)
    #app.image_rec.evaluate()
    main_loop = threading.Thread(target=root.mainloop())
    return app, main_loop
    
    

if __name__ == "__main__":
    start_app()