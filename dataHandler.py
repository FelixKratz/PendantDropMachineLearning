import os.path
import os
import re
import numpy as np
from sklearn.utils import shuffle

class DataHandler:
    def __init__(self, preprocessedDataFile=""):
        self.data = []
        self.labels = []
        self.volume = []

        if not preprocessedDataFile == "":
            self.loadDataFromPreprocessedFile(preprocessedDataFile)

    def saveData(self, file):
        np.savez(file, data=self.data, labels=self.labels)

    def append(self, dataHandler):
        self.data = np.concatenate((self.data, dataHandler.data))
        self.labels = np.concatenate((self.labels, dataHandler.labels))
        self.volume = np.concatenate((self.volume, dataHandler.volume))

    def shuffle(self):
        self.data, self.labels, self.volume = shuffle(self.data, self.labels, self.volume)

    def zeroPadData(self, newLength):
        for i in range(len(self.data)):
            nonZeroLength = len(self.data.T[i].T[0])
            if nonZeroLength > newLength:
                nonZeroLength = newLength

            tmp = np.copy(self.data.T[i])
            self.data.T[i] = np.zeros((newLength,2))
            self.data.T[i][:nonZeroLength] = tmp[:nonZeroLength]

        self.data = np.stack(self.data)

    def reshapeData(self):
        self.data = self.data.reshape(len(self.data), len(self.data[0]) * len(self.data[0][0]))

    # readDataPercentModes:
    #                       "front" => read the first "readDataPercentage" percent from the disk
    #                       "back"  => read the last "readDataPercentage" percent from the disk
    # Can be used to train and evaluate on first 60% and test on last 40% of the data
    def loadDataFromPreprocessedFile(self, file, readDataPercentage=1., readDataPercentMode="front"):
        dat = np.load(file)
        self.data = dat['data']
        self.labels = dat['labels']

        if (readDataPercentMode == "front"):
            self.data = self.data[:int(len(self.data) * readDataPercentage)]
            self.labels = self.labels[:int(len(self.labels) * readDataPercentage)]
        elif (readDataPercentMode == "back"):
            self.data = self.data[int(len(self.data) * (1 - readDataPercentage)):]
            self.labels = self.labels[int(len(self.labels) * (1 - readDataPercentage)):]
        else:
            print("Error: Invalid Data Percent Mode!")
            exit(1)

    def calculateVolume(self):
        self.volume = []
        for i in range(len(self.data)):
            r = self.data[i].T[0]
            z = self.data[i].T[1]

            V = np.pi * np.trapz(x=z, y=r**2)
            self.volume.append(V)
        self.volume = np.array(self.volume)

    def appendVolumeToData(self):
        appendedData = []
        for i in range(len(self.data)):
            appendedData.append(np.append(self.data[i], self.volume[i]))
        data = np.stack(appendedData)
        self.data = data
