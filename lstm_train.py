import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import sys

from prepare_rolling_prices import prepare_rolling_prices

def trainLSTM():
	training = pd.read_csv("./training_files/training.csv")
	x_train, y_train = prepare_rolling_prices(training)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	testing = pd.read_csv("./training_files/testing.csv")
	x_test, y_test = prepare_rolling_prices(training)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	print("Data Preparation Completed")

	# create Model
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))  # 2 time step, 1 feature
	model.add(LSTM(50, return_sequences = False))
	model.add(Dense(25))
	model.add(Dense(1)) # 1 output: Price

	# Train
	epochs = 100
	train_scores = []
	test_scores = []
	train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
	earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
	model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
	test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(model.evaluate(x_test, y_test)[0]))
	model.fit(x_train, y_train, batch_size=2000, epochs=epochs, callbacks=[train_loss, test_loss, earlystopper])
	model.save('./models/my_model.h5')
	print("Training Completed")

if __name__ == '__main__':
	trainLSTM()