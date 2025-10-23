import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pathlib import Path
from os import getcwd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from abc import ABC


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SupportVectorClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        """
        Initializes the Support Vector Classifier model.
        
        Args:
            kernel (str): Specifies the kernel type to be used in the algorithm.
            C (float): Regularization parameter.
        """
        self.model = SVC(kernel=kernel, C=C, gamma='scale', probability=True)
        print("SVC Model Initialized.")

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

    def predict(self, X_test):
        """
        Makes a prediction on the given test data.
        
        Args:
            X_test (array-like): Test data features (a single flattened image).
        
        Returns:
            The predicted label.
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's accuracy on the test set.
        
        Args:
            X_test (array-like): Test data features.
            y_test (array-like): True labels for the test data.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"SVC Accuracy: {accuracy * 100:.2f}%")


#MLP model
class MLPnn(nn.Module):
    def __init__(self, in_dim=28*28, hidden1=256, hidden2=128, num_classes=10, p=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return F.log_softmax(logits, dim=1)
