import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pathlib import Path
from os import makedirs
from enum import Enum


# ======================
# MODEL DEFINITIONS
# ======================

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.dropout(x)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class MLPnn(nn.Module):
    def __init__(self, in_dim=28*28, hidden1=256, hidden2=128, num_classes=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)


# ======================
# BASE CLASS
# ======================

DATA_FOLDER = Path("ImageRecognition", "data")
MODEL_FOLDER = Path("ImageRecognition", "model")


class ModelBase:
    def __init__(self, batch_size=64, model_path=None):
        print("Initializing dataloaders...")
        self.__initialize_dataloaders__(batch_size)
        print("Initializing model...")
        self.__initialize_model__(model_path)
        print("Initialization complete.")

    def __initialize_dataloaders__(self, batch_size):
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        self.training_data = MNIST(DATA_FOLDER, train=True, download=True, transform=transform)
        self.test_data = MNIST(DATA_FOLDER, train=False, download=True, transform=transform)
        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

    def __initialize_model__(self, model_path):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError


# ======================
# CNN MODEL
# ======================

class ModelCNN(ModelBase):
    def __initialize_model__(self, model_path=None):
        makedirs(MODEL_FOLDER, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        if model_path and Path(model_path).exists():
            print(f"Loading CNN model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        print(self.model)

    def train(self):
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % 100 == 0:
                print(f"Batch {batch}: loss = {loss.item():.4f}")

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(self.test_dataloader)
        accuracy = 100 * correct / len(self.test_data)
        print(f"Test Accuracy: {accuracy:.2f}%, Avg loss: {test_loss:.4f}")

    def save_model(self):
        save_path = Path(MODEL_FOLDER, "cnn_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"CNN model saved to {save_path}")

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            pred = self.model(input)
            return pred.argmax(1).item()


# ======================
# SVC MODEL
# ======================

class ModelSVC(ModelBase):
    def __initialize_model__(self, model_path=None):
        print("Preparing data for SVC...")
        self.X_train = self.training_data.data.numpy().reshape(len(self.training_data), -1)
        self.y_train = self.training_data.targets.numpy()
        self.X_test = self.test_data.data.numpy().reshape(len(self.test_data), -1)
        self.y_test = self.test_data.targets.numpy()

        self.model = SVC(kernel="rbf", C=1.0, gamma="scale")
        print("SVC initialized.")

    def train(self):
        print("Training SVC model...")
        self.model.fit(self.X_train, self.y_train)
        print("SVC training complete.")

    def test(self):
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"SVC Test Accuracy: {acc*100:.2f}%")

    def save_model(self):
        import joblib
        save_path = Path(MODEL_FOLDER, "svc_model.pkl")
        joblib.dump(self.model, save_path)
        print(f"SVC model saved to {save_path}")

    def predict(self, input):
        input_flat = input.reshape(1, -1)
        return self.model.predict(input_flat)[0]


# ======================
# MLP MODEL
# ======================

class ModelMLP(ModelBase):
    def __initialize_model__(self, model_path=None):
        makedirs(MODEL_FOLDER, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPnn().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        if model_path and Path(model_path).exists():
            print(f"Loading MLP model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        print(self.model)

    def train(self):
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % 100 == 0:
                print(f"Batch {batch}: loss = {loss.item():.4f}")

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(self.test_dataloader)
        accuracy = 100 * correct / len(self.test_data)
        print(f"MLP Test Accuracy: {accuracy:.2f}%, Avg loss: {test_loss:.4f}")

    def save_model(self):
        save_path = Path(MODEL_FOLDER, "mlp_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"MLP model saved to {save_path}")

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            pred = self.model(input)
            return pred.argmax(1).item()


# ======================
# TESTING
# ======================

if __name__ == "__main__":
    # Example usage
    print("Testing CNN...")
    cnn = ModelCNN()
    cnn.train()
    cnn.test()
    cnn.save_model()

    print("Testing MLP...")
    mlp = ModelMLP()
    mlp.train()
    mlp.test()
    mlp.save_model()

    print("Testing SVC...")
    svc = ModelSVC()
    svc.train()
    svc.test()
    svc.save_model()
