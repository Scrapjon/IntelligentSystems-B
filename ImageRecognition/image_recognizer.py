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

    def __init__(self, batch_size=64, model_path=None):
        self.model_type = ModelType.CNN # Default

        self.models: dict[ModelType, ModelBase] = {
            ModelType.CNN: ModelCNN(batch_size=batch_size, model_path=model_path),
            ModelType.MLP: ModelMLP(batch_size=batch_size, model_path=model_path),
            ModelType.SVC: ModelSVC(batch_size=batch_size, model_path=model_path)
        }

    def train(self):
        for model in self.models.values():
            model.train()

    def test(self):
        for model in self.models.values():
            model.test()

    def save_model(self):
        for model in self.models.values():
            model.save_model()

    def predict(self, model_type, input):
        return self.models[model_type].predict(input)

