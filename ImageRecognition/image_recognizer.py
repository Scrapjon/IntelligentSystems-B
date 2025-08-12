from ImageRecognition.neural_network import NeuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pathlib import Path
from os import getcwd

DATA_FOLDER = Path("ImageRecognition","data")
MODEL_FOLDER = Path("ImageRecognition", "model")

class ImageRecognizer():
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
        
        print("Initializing Neural Network")
        self.__initialize_neural_network__(model_path=model_path)
        print("Initialized Neural Network")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        

    def __initialize_dataloaders__(self,batch_size):
        self.batch_size = batch_size
        self.training_data = MNIST(DATA_FOLDER, train=True, download=True, transform=ToTensor())
        self.test_data = MNIST(DATA_FOLDER, train=False, download=True, transform=ToTensor())
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
    
    def __initialize_neural_network__(self, model_path:Path = None):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = NeuralNetwork().to(self.device)
        if model_path:
            if model_path.exists():
                print(f"Loading model at path: {model_path}")
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
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
        classes = self.test_data.classes
        self.model.eval()
        x, y = self.test_data[0][0], self.test_data[0][1]
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')



            

    
       
        
    

if __name__ == "__main__":
    imgrec = ImageRecognizer()
