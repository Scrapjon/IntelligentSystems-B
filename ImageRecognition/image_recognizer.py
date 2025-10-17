from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from pathlib import Path
from os import getcwd, makedirs
from enum import Enum

if __name__ != "__main__":
    from ImageRecognition.models import NeuralNetwork
else:
    from models import NeuralNetwork


class ModelType(Enum):
    CNN = 0
    SVC = 1
    MLP = 2

DATA_FOLDER = Path("ImageRecognition","data")
MODEL_FOLDER = Path("ImageRecognition", "model")

class ImageRecognizer():
    def __init__(self, model_path = None, model_type = ModelType.CNN):
        self.model_type = model_type
        
        if self.model_type == ModelType.CNN:
            print("Initializing Neural Network")
            self.__initialize_neural_network__(model_path=model_path)
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
            print("Initialized Neural Network")
        elif self.model_type == ModelType.SVC:
            self.__initialize_SVC()
        else:
            raise ValueError("Invalid model_type specified. Choose 'nn' or 'svc'.")
        
class ModelBase:
    def __init__(self, batch_size = 64, model_path = None):

        """
        Initializes the ImageRecognizer object. Passing in model_path loads the model from a file

        """
        

        print("Initializing Dataloaders")
        self.__initialize_dataloaders__(batch_size)

        print("Initialized Dataloaders")
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        
        print("Initializing Model")
        self.__initialize_model__(model_path=model_path)
        print("Initialized Model")

        
        
        
        #old version below
        '''print("Initializing Neural Network")
        self.__initialize_neural_network__(model_path=model_path)
        print("Initialized Neural Network")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)'''

        

    def __initialize_dataloaders__(self,batch_size):
        self.batch_size = batch_size
        self.transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
        ])
        self.training_data = MNIST(DATA_FOLDER, train=True, download=True, transform=self.transform)
        self.test_data = MNIST(DATA_FOLDER, train=False, download=True, transform=self.transform)
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
        pass
    
    def __initialize_model__(self, model_path:Path = None):
        pass

    def train(self, X_train = None, y_train = None):
        pass
        

    def test(self, X_test = None, y_test = None):
        pass
    
    def training_loop(self,epochs):
        pass
        
    
    def save_model(self):
        pass

    def predict(self, input):
        pass


class ModelCNN(ModelBase):
    def __init__(self, batch_size=64, model_path=None):
        super().__init__(batch_size, model_path)

    def __initialize_dataloaders__(self, batch_size):
        return super().__initialize_dataloaders__(batch_size)
    
    def __initialize_model__(self, model_path = None):
        makedirs(MODEL_FOLDER, exist_ok=True)

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = NeuralNetwork().to(self.device)
        if model_path:
            if model_path.exists():
                print(f"Loading model at path: {model_path}")
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
        print(self.model)

    def train(self, X_train = None, y_train = None):
        """
        X_train and y_train are ignored to maintain API consistancy
        """
    def train(self, X_train = None, y_train = None):
        """
        X_train and y_train are ignored to maintain API consistancy
        """
        dataloader = self.train_dataloader
        model = self.model
        device = self.device
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        size = len(dataloader)
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, X_test = None, y_test = None):
        """
        X_test and y_test are ignored to maintain API consistancy
        """
    def test(self, X_test = None, y_test = None):
        """
        X_test and y_test are ignored to maintain API consistancy
        """
        dataloader = self.test_dataloader
        model = self.model
        device = self.device
        loss_fn = self.loss_fn


        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.model = model # just in case i forgot to reassign

    def training_loop(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")


    def save_model(self):
        save_path = Path(MODEL_FOLDER, "model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model state to {save_path}")

    def predict(self, input):
        """
        TODO: add in logic to take in images rather than just encoded data.
        """
        classes = self.test_data.classes
        self.model.eval()
        with torch.no_grad():
            x = input.to(self.device)
            pred = self.model(x)
            predicted  = classes[pred[0].argmax(0)]
            return predicted

class ModelSVC(ModelBase):
    def __init__(self, batch_size=64, model_path=None):
        super().__init__(batch_size, model_path)

    def __initialize_model__(self, model_path = None):
        """
        Prepares data, initializes, and trains the Support Vector Classifier.
        """
        print("Initializing SVC...")
        makedirs(MODEL_FOLDER, exist_ok=True)
        
        # 1. Extract data from the PyTorch datasets and reshape it for scikit-learn
        # Training data
        self.train_images = self.training_data.data.numpy()
        self.train_labels = self.training_data.targets.numpy()
        self.n_train_samples = len(self.train_images)
        self.X_train = self.train_images.reshape((self.n_train_samples, -1)) # Flatten images to 784-dim vectors
    
        # Testing data (for evaluation)
        self.test_images = self.test_data.data.numpy()
        self.test_labels = self.test_data.targets.numpy()
        n_test_samples = len(self.test_images)
        self.X_test = self.test_images.reshape((n_test_samples, -1))

        print(f"Data prepared for SVC: {self.n_train_samples} training samples, {n_test_samples} test samples.")

        # 2. Initialize SVC model
        self.model = SVC(kernel="rbf", C=1.0, gamma='scale', probability=True)

    def __initialize_dataloaders__(self, batch_size):
        return super().__initialize_dataloaders__(batch_size)


    def train(self, X_train, y_train):
        """
        Trains the SVC model on the provided training data.
        
        Args:
            X_train (array-like): Training data features (flattened images).
            y_train (array-like): Training data labels.
        """
        print("Training SVC model...")
        self.model.fit(X_train, y_train)
        print("SVC Training complete.")

    def test(self, X_test, y_test):
        """
        Evaluates the model's accuracy on the test set.
        
        Args:
            X_test (array-like): Test data features.
            y_test (array-like): True labels for the test data.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"SVC Accuracy: {accuracy * 100:.2f}%")


class ModelSVC(ModelBase):
    def __init__(self, batch_size=64, model_path=None):
        super().__init__(batch_size, model_path)

    def __initialize_model__(self, model_path = None):
        """
        Prepares data, initializes, and trains the Support Vector Classifier.
        """
        print("Initializing SVC...")
        makedirs(MODEL_FOLDER, exist_ok=True)
        
        # 1. Extract data from the PyTorch datasets and reshape it for scikit-learn
        # Training data
        self.train_images = self.training_data.data.numpy()
        self.train_labels = self.training_data.targets.numpy()
        self.n_train_samples = len(self.train_images)
        self.X_train = self.train_images.reshape((self.n_train_samples, -1)) # Flatten images to 784-dim vectors
        self.y_train = self.train_labels
    
        # Testing data (for evaluation)
        self.test_images = self.test_data.data.numpy()
        self.test_labels = self.test_data.targets.numpy()
        n_test_samples = len(self.test_images)
        self.X_test = self.test_images.reshape((n_test_samples, -1))
        self.y_test = self.test_labels

        print(f"Data prepared for SVC: {self.n_train_samples} training samples, {n_test_samples} test samples.")

        # 2. Initialize SVC model
        self.model = SVC(kernel="rbf", C=1.0, gamma='scale', probability=True, verbose=True)


    def train(self, X_train = None, y_train = None):
        """
        Trains the SVC model on the provided training data.
        
        Args:
            X_train (array-like): Training data features (flattened images).
            y_train (array-like): Training data labels.
        """
        if X_train == None:
            X_train = self.X_train
        if y_train == None:
            y_train = self.y_train
        print("Training SVC model...")
        self.model.fit(X_train, y_train)
        print("SVC Training complete.")

    def test(self, X_test, y_test):
        """
        Evaluates the model's accuracy on the test set.
        
        Args:
            X_test (array-like): Test data features.
            y_test (array-like): True labels for the test data.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"SVC Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    svc = ModelSVC()
    svc.train()
