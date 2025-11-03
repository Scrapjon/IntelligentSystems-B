from pathlib import Path
from gui import DigitDrawingApp, start_app, ModelType
from ImageRecognition.models import evaluate_models
if __name__ == "__main__":
    print("Starting app...")
    app, app_thread = start_app()
    try:
        with open("result_doc.txt", "w", encoding="utf-8") as f:
            results = ("#"*5)+"TESTING"+("#"*5)+("\n"*3)
            print(results)
            for accuracy, model_type in app.run_tests():
                to_add = f"{model_type}: {accuracy}% accuracy\n\n"
                print(to_add)
                results += to_add
            evaluation_header = ("#"*5)+"EVALUATION"+("#"*5)+("\n"*3)
            print(evaluation_header)
            for result in evaluate_models({
                "cnn": app.image_rec.models[ModelType.CNN],
                "mlp": app.image_rec.models[ModelType.MLP],
                "svc": app.image_rec.models[ModelType.SVC]
                }):
                print(result)
                results += result
            f.write(results)

            
        
    except Exception as e:
        print(e)
    app.root.mainloop()
