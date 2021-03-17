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
	x_test, y_test = prepare_rolling_prices(testing)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	print("Data Preparation Completed")
	epochs = 500
	batch_size = 10

	# create Model
	model = Sequential()
	model.add(LSTM(64, return_sequences=True, stateful=True, batch_input_shape = (batch_size,x_train.shape[1], 1)))  # 2 time step, 1 feature
	model.add(LSTM(64, return_sequences = True))
	model.add(LSTM(64))
	model.add(Dense(20))
	model.add(Dense(1)) # 1 output: Price

	# Train
	train_scores = []
	test_scores = []
	train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
	earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
	model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
	test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(model.evaluate(x_test, y_test, batch_size=batch_size)[0]))
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[train_loss, test_loss, earlystopper])
	result = model.evaluate(x_test, y_test, batch_size=batch_size)[1]
	model.save('./models/my_model_4.h5')
	print("Training Completed")

	plt.figure()

	plt.grid()
	plt.title("Testing RMSE: " + str(result))
	plt.suptitle("Learning Curve")
	plt.ylabel("loss")
	plt.xlabel("epochs")
	plt.ylim(top=max(train_scores), bottom=min(train_scores))
	plt.plot(np.linspace(0, len(train_scores), len(train_scores)), train_scores, linewidth=1, color="r",
			 label="Training loss")
	plt.plot(np.linspace(0, len(test_scores), len(test_scores)), test_scores, linewidth=1, color="b",
			 label="Testing loss")
	legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
	legend.get_frame().set_facecolor('C0')

	plt.show()

if __name__ == '__main__':
	trainLSTM()