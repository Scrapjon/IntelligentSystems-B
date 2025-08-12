import ImageRecognition
from ImageRecognition.image_recognizer import ImageRecognizer
from pathlib import Path

EPOCHS = 5
EVAL_LOOPS = 10
MODEL_PATH = Path("ImageRecognition", "model", "model.pth")
CLASSES = ["",""]



### Training and save ###
imgrec_train_demo = ImageRecognizer()

imgrec_train_demo.training_loop(EPOCHS)
imgrec_train_demo.save_model()

### Loading and Evaluation ###
imgrec_test_demo = ImageRecognizer(model_path=MODEL_PATH)

for i in range(EVAL_LOOPS):
    imgrec_test_demo.evaluate()
