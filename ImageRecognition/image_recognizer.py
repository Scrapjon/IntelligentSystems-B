if __name__ != "__main__": # Relative imports
    from .models import *
else:
    from models import *



# ======================================================
# === IMAGE RECOGNIZER WRAPPER CLASS
# ======================================================

class ModelType(Enum):
    CNN = "cnn"
    MLP = "mlp"
    SVC = "svc"


class ImageRecognizer:
    """Unified interface for all models."""

    def __init__(self, batch_size=64, model_dir=None): # <-- Changed to model_dir
        self.model_type = ModelType.CNN # Default

        # --- START FIX: Construct specific paths for each model ---
        cnn_path = model_dir / "cnn_model.pth" if model_dir else None
        mlp_path = model_dir / "mlp_model.pth" if model_dir else None
        svc_path = model_dir / "svc_model.pkl" if model_dir else None
        # --- END FIX ---

        self.models: dict[ModelType, ModelBase] = {
            # Pass the correct path to each constructor
            ModelType.CNN: ModelCNN(batch_size=batch_size, model_path=cnn_path),
            ModelType.MLP: ModelMLP(batch_size=batch_size, model_path=mlp_path),
            ModelType.SVC: ModelSVC(batch_size=batch_size, model_path=svc_path)
        }

    def train(self):
        # Add epochs to SVC train call
        self.models[ModelType.CNN].train()
        self.models[ModelType.MLP].train()
        self.models[ModelType.SVC].train() # SVC train doesn't use epochs, but call is fine

    def test(self):
        for model in self.models.values():
            model.test()

    def save_model(self):
        for model in self.models.values():
            model.save_model()

    def predict(self, model_type, input):
        return self.models[model_type].predict(input)