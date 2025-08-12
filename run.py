import ImageRecognition
from ImageRecognition.image_recognizer import ImageRecognizer
from pathlib import Path

EPOCHS = 5
MODEL_PATH = Path("ImageRecognition", "model", "model.pth")

imgrec = ImageRecognizer(model_path=MODEL_PATH)

imgrec.training_loop(EPOCHS)
imgrec.save_model()