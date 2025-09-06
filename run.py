import ImageRecognition
from ImageRecognition.image_recognizer import ImageRecognizer
from pathlib import Path

EPOCHS = 5
MODEL_PATH = Path("ImageRecognition", "model", "model.pth")
CLASSES = ["",""]

TRAINING_MODE = True


if TRAINING_MODE:
    ### Training and save ###
    imgrec_train_demo = ImageRecognizer()

    imgrec_train_demo.training_loop(EPOCHS)
    imgrec_train_demo.save_model()
try:
    ### Loading and Evaluation ###
    imgrec_test_demo = ImageRecognizer(model_path=MODEL_PATH)
    imgrec_test_demo.test_data
    import random
    correct = []
    classes = imgrec_test_demo.test_data.classes
    for i in range(EVAL_LOOPS):
        pred = imgrec_test_demo.predict(imgrec_test_demo.test_data[i-1][0])
        actual = classes[imgrec_test_demo.test_data[i-1][1]]
        print(f"prediction: {pred}, actual: {actual}")
        if pred == classes[imgrec_test_demo.test_data[i-1][1]]:
            
            correct.append(pred)
    accuracy = len(correct)/EVAL_LOOPS
    print(f"Accuracy: {accuracy*100}%\nScore: {len(correct)}/{EVAL_LOOPS}")
except Exception as e:
    print("Failed to evaluate due to:",e)