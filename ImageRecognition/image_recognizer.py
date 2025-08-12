from ImageRecognition.neural_network import NeuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pathlib import Path
from os import getcwd

DATA_FOLDER = Path("ImageRecognition","data")

class ImageRecognizer():
    def __init__(self, batch_size = 64):
        print("Initializing Dataloaders")
        self.__initialize_dataloaders__(batch_size)

        print("Initialized Dataloaders")
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        
        print("Initializing Neural Network")
        self.__initialize_neural_network__()
        print("Initialized Neural Network")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        

    def __initialize_dataloaders__(self,batch_size):
        self.batch_size = batch_size
        self.training_data = MNIST(DATA_FOLDER, train=True, download=True, transform=ToTensor())
        self.test_data = MNIST(DATA_FOLDER, train=False, download=True, transform=ToTensor())
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
    
    def __initialize_neural_network__(self):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = NeuralNetwork().to(self.device)
        print(self.model)
    
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
    
    def start_training(self,epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")
            

    
       
        
    

if __name__ == "__main__":
    imgrec = ImageRecognizer()
