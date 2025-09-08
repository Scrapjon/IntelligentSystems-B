import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk, ImageOps
import numpy as np
import cv2
import threading, asyncio
import ImageRecognition
from pathlib import Path
from ImageRecognition.image_recognizer import ImageRecognizer
import torch

MODEL_PATH = Path(__file__, "ImageRecognition", "model", "model.pth")

class DigitDrawingApp:
    def __init__(self, root, model_path = None):
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

        self.train_btn = ttk.Button(root, text="Train", command=self.train)
        self.train_btn.grid(row=1,column=3, pady=10)

        self.epochs = tk.IntVar(root, value=5)

        self.train_entry = ttk.Entry(root, textvariable=self.epochs)
        self.train_entry.grid(row=1,column=4, pady=10)



        # Panel for processed images
        self.processed_frame = tk.Frame(root)
        self.processed_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Labels for showing processed images
        self.gray_label = tk.Label(self.processed_frame)
        self.gray_label.pack(side='left', padx=5)

        self.binary_label = tk.Label(self.processed_frame)
        self.binary_label.pack(side='left', padx=5)

        self.edge_label = tk.Label(self.processed_frame)
        self.edge_label.pack(side='left', padx=5)

        # For drawing
        self.old_x = None
        self.old_y = None
        self.pen_width = 8  # thickness of the drawing pen
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Create a PIL image to draw on
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), 'white')
        self.draw_image = ImageDraw.Draw(self.image)

        self.image_rec = ImageRecognizer(64, model_path)

    def train(self):
        epochs = self.epochs.get() if self.epochs.get() > 0 else 1
        def train_sequence(self,epochs):
            self.image_rec.training_loop(epochs)
            self.image_rec.save_model()
        training_thread = threading.Thread(target=train_sequence, args=(self,epochs))
        training_thread.start()

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
        img_np = img_np.max()-img_np

        # Normalise
        normalised_img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())

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

        return {
            "grey": gray,
            "img": img_np,
            "normalised": normalised_img_np,
            "edges": edges
        }

    def predict_drawing(self):
        def prediction_sequence(self):
            drawing = self.process_drawing()
            normalised = drawing["normalised"]
            normalised =  np.array([normalised])
            normalised = torch.as_tensor(data=normalised,dtype=torch.float,device=self.image_rec.device)
            
            pred = self.image_rec.predict(normalised)

            return pred
        prediction_thread = threading.Thread(target=prediction_sequence, args=[self])
        prediction_thread.start()
        prediction = prediction_thread.join()
        
        return prediction
        




if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawingApp(root, MODEL_PATH)
    app.image_rec.evaluate()
    root.mainloop()
