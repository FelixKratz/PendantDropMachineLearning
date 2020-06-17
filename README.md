Developed at TU Dortmund University

Author: Felix Kratz (felix.kratz@tu-dortmund.de)<br/>
Supervisor: Jan Kierfeld (jan.kierfeld@tu-dortmund.de)

# Citation and use
If you use this work in your research department or company and
you publish data obtained with this, please cite the following
articles:

* Following soon



# Use of the provided neural nets

## Prerequisites:
The provided neural nets need [Tensorflow 2](https://www.tensorflow.org/install "Tensorflow Install Instructions")
to be installed on the machine. **Keras** is now part of **Tensorflow** such that
no additional installation is needed.

An example of the instructions below is provided in the form of
the jupyter notebook "example.ipynb". Also provided are some
example shapes generated numerically from the Young-Laplace
equation.

## Loading the neural net
The neural net can be loaded by using the provided ***AIMan*** module by
performing the command while exchanging "model_xy.h5" for the
name of the desired network

```python
aiMan = AIMan("models/model_xy.h5")
```
## Format of the data
The provided ***DataHandler*** can be used to save an load the data in an
optimized and efficient way. To save data using the ***DataHandler***
following code can be used
```python
dataHandler = DataHandler()
dataHandler.data = coordinates
dataHandler.saveData("path/to/file")
```
where **coordinates** is an array of the evenly sampled shape coorinates
with an arclength step of 1e-2.

To load data from the preprocessed file the following code can be used when
initializing a new ***DataHandler*** object
```python
dataHandler = DataHandler("path/to/preprocessed/file")
```
**or alternatively**, if a ***DataHandler*** object **dataHandler** is already
present:
```python
dataHandler.loadDataFromPreprocessedFile("path/to/preprocessed/file",
                                          readDataPercentage=1.,
                                          readDataPercentMode="front")
```
where additional arguments can be used an from which end ("front", "back") the
data is read and which percentage of the data should be read (1: all, 0: none).

### Networks sampled uniformly in  p<sub>L</sub> - &Delta; &rho; space
This applies to the networks model_uniform_rho.h5 and model_uniform_rho_noise.h5.

The input is a vector of 452 elements per batch, hence the input to the
network has shape (batches, 452) which consists of 226 dimensionless
coordinate pairs that are spaced evenly with an arclength step of 1e-2, when
given the zero padded coordinate data **zeroPaddedCoordinates** in shape zeroPaddedCoordinates.shape = (batches, 226, 2), can be formated by using
the ***reshape*** function from ***numpy***
```python
inputData = zeroPaddedCoordinates.reshape(len(zeroPaddedCoordinates),
                                          len(zeroPaddedCoordinates[0]) * len(zeroPaddedCoordinates[0][0]))
```

**or alternatively**, using the provided ***DataHandler***
```python
dataHandler = DataHandler()
dataHandler.data = coordinates
dataHandler.zeroPadData(newLength=226)
dataHandler.reshapeData()
```

### Network sampled uniformly in  p<sub>L</sub> - Wo space
This applies to the network model_uniform_Wo.h5.

The input is a vector that consists of 1025 elements per batch,
hence the input to the network has the shape (batches, 1025).
The input vector is build up by 512 coordinate pairs and the
dimensionless volume of the drop.
If the data is present in the shape
coorinates.shape = (batches, 512, 2); volume.shape = (batches)
the input vector can be constructed by using following command
using numpy as np:
```python
inputData = coorinates.reshape(len(zeroPaddedCoordinates), len(zeroPaddedCoordinates[0]) * len(zeroPaddedCoordinates[0][0]))

appendedData = []
for i in range(len(zeroPaddedCoordinates)):
    appendedData.append(np.append(zeroPaddedCoordinates[i], volume[i]))
inputData = np.stack(appendedData)
```

**or alternatively**, using the provided ***DataHandler***
```python
dataHandler = DataHandler()
dataHandler.data = coordinates
dataHandler.calculateVolume()
dataHandler.zeroPadData(newLength=512)
dataHandler.reshapeData()
dataHandler.appendVolumeToData()
```

## Predict material parameters
To predict the dimensionless control parameters of the Young-Laplace equation
with the loaded neural net following command can be used:

```python
parameters = aiMan.predict(inputData)
```

**or**, when using a ***DataHandler*** object **dataHandler**

```python
parameters = aiMan.predict(dataHandler.data)
```

where parameters has shape (batches, 2). For shape i the control parameters
can then be accessed via

```python
p = parameters[i][0]
rho = parameters[i][0]
```

## Outlook
An image preprocessing frontend has been developed, but is still
in an experimental phase. It will be added to this repo as soon as it has been tested extensively.
