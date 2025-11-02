from pathlib import Path
from gui import DigitDrawingApp, start_app

if __name__ == "__main__":
    app, app_thread = start_app()
    try:
        for accuracy, model_type in app.run_tests():
            print(f"{model_type}: {accuracy}% accuracy")
    except Exception as e:
        print(e)
    app.root.mainloop()
