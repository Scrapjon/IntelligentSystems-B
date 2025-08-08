from torchvision.datasets import MNIST
from pathlib import Path
from os import getcwd

DATA_FOLDER = Path("ImageRecognition","data")

dataset = MNIST(DATA_FOLDER, download=True)
