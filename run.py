from pathlib import Path
from gui import DigitDrawingApp, start_app, ModelType
from ImageRecognition.models import evaluate_models

if __name__ == "__main__":
    print("Starting app...")
    app, app_thread = start_app()
    app.root.mainloop()
