from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

    