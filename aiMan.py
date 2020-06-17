import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model

class AIMan:
    def __init__(self, file=""):
        self.model = None
        self.guess = None
        if not file == "":
            self.loadModel(file=file)

    def loadModel(self, file):
        if os.path.isfile(file):
            tf.keras.backend.clear_session()
            self.model = load_model(file)
            return True
        else:
            print("Warning: Model file does not exist!")
            return False

    def predict(self, data):
        if len(data) == 0:
            print("Error: Can not predict because input data is missing!")
            exit(1)

        self.guess = self.model.predict(data)
        return self.guess
