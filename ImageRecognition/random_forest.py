import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.ensemble import RandomForestClassifier

def train(train: Tensor, test: Tensor):
    print("RANDOM FOREST!")
    train_df = pd.DataFrame(train.detach().numpy())
    test_df = pd.DataFrame(test.detach().numpy())

    print(train_df)
    print(test_df)
    


