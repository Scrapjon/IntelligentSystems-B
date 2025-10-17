from ImageRecognition.models import NeuralNetwork, SupportVectorClassifier
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from pathlib import Path
from os import getcwd, makedirs
from enum import Enum

class ModelType(Enum):
    CNN = 0
    SVC = 1
    MLP = 2

DATA_FOLDER = Path("ImageRecognition","data")
MODEL_FOLDER = Path("ImageRecognition", "model")

class ImageRecogniser():
    def __init__(self, model_path = None, model_type = ModelType.CNN):
        self.model_type = model_type



class CNN():
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
    
    def __initialize_neural_network__(self, model_path:Path = None):

        makedirs(MODEL_FOLDER, exist_ok=True)

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = NeuralNetwork().to(self.device)
        if model_path:
            if model_path.exists():
                print(f"Loading model at path: {model_path}")
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
        print(self.model)

    def __initialize_SVC(self, train = False):
        """
        Prepares data, initializes, and trains the Support Vector Classifier.
        """
        print("Initializing SVC...")
        
        # 1. Extract data from the PyTorch datasets and reshape it for scikit-learn
        # Training data
        train_images = self.training_data.data.numpy()
        train_labels = self.training_data.targets.numpy()
        n_train_samples = len(train_images)
        X_train = train_images.reshape((n_train_samples, -1)) # Flatten images to 784-dim vectors
    
        # Testing data (for evaluation)
        test_images = self.test_data.data.numpy()
        test_labels = self.test_data.targets.numpy()
        n_test_samples = len(test_images)
        X_test = test_images.reshape((n_test_samples, -1))

        print(f"Data prepared for SVC: {n_train_samples} training samples, {n_test_samples} test samples.")

        # 2. Initialize your SVC model from Ml_models.py
        self.svc_model = SupportVectorClassifier()

        # 3. Train the model
        if train:
            self.svc_model.train(X_train, train_labels)

            # 4. (Optional) Evaluate the model immediately after training
            self.svc_model.evaluate(X_test, test_labels)
            print("SVC is ready.")
    
    def train(self):

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

        

    
    def test(self):
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
    
    def training_loop(self,epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")
    
    def save_model(self):
        save_path = Path(MODEL_FOLDER, "model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model state to {save_path}")

    def evaluate(self):
        import random
        index = random.randint(0,len(self.test_data)-1)
        classes = self.test_data.classes
        self.model.eval()
        x, y = self.test_data[index][0], self.test_data[index][1]
        print(type(x))
        import time
        for i in range(len(torch.detach_copy(x))):
            time.sleep(0.5)
            print(torch.detach_copy(x)[i])
            print(i)
        print(self.test_data)
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
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
        

if __name__ == "__main__":
    imgrec = ImageRecognizer()
    imgrec.evaluate()
