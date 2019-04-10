# Tensorflow Implementation of Convolutional Autoencoder
### Project details
**Autoencoder** is used as powerful tool to extract important features from the input data and then use the encoded (latent) represntation as a new input for classification model. The nice thing about autoencoder is that we don't need labels to train it and it belong to **unsupervised learning** technically. In simple words, it does the *dimensionality reduction* while preserving the vital features that are transformed into a *latent space*. For an image classification/detection task, a particular type of autoencoder called **convolutional autoencoder** (CAE) is used.

This project demonstrates how CAE can be implemented in tensorflow framework. It also contains a sample pretrained model which is trained on **Fashion-MNIST Dataset**.

### Getting started
1. Make sure `python 3.x` is installed
2. To install other dependencies:
```
pip install -r requirements.txt
``` 
3. To test a pretrained model (like examples in out folder) by feeding a random image from validation set, run 
```
python autoencoder.py test -m ./out/Training__20190407_154622/autoencoder_latest.meta
```
4. To train a new model, run 
```
python autoencoder.py train
```
5. To continue training a pretrained model, run 
```
python autoencoder.py train -m ./out/Training__20190407_154622/autoencoder_latest.meta
```
6. To get these details in terminal, simply execute
```
python autoencoder.py -h
```

### Results
1. When the pretrained model is tested, you should see the output something like this
![result](./images/test_result.png)
2. The training and validation loss convergence of the trained model is shown below:
![training](./images/training.png)

### Instructions
1. **data** folder contains the *fashion-MNIST dataset* which can be replaced with custom dataset as well.
2. **out** folder gets automatically created when the model is getting trained. It contains the latest and the best model checkpoints. It also stores the log necessary for tensorboard visualization.
3. **autoencoder.py** contians the main code for training and testing the model.
4. **autoencoder_model_db.py** file contains some example model architectures of CAE. Please modify or add new definitions here. Please esure that `autoencoder.py` file is modified accordingly to call the new function.
5. **fashion_mnist_utils.py** contains the functions related to pre-processing and loading of *fashion-MNIST dataset*.
6. To see the training progress/summary and also compare between different models, use **tensorboard** for visualization by running
```
tensorboard --logdir='./out/'
```
