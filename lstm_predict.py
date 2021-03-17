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
import json
from matplotlib import pyplot as plt

def lstm_predict(ID, name):
	model = load_model('./models/my_model.h5')
	min_price, max_price, mean_price = -1, -1, -1
	with open("./utils/min_max_mean.txt", "r") as f:
		line = str(f.read())
		nums = line.split(",")
		min_price = int(nums[0])
		max_price = int(nums[1])
		mean_price = float(nums[2])

	print(min_price, max_price, mean_price)


	testing = pd.read_csv("./training_files/training.csv")
	this_df = testing.loc[testing['ID'] == int(ID)]["price"]
	this_df = this_df.astype(int)

	this_df = min_max_normalize(this_df, min_price, max_price, mean_price)
	this_np_array = np.array(this_df)

	# Get Last 30 days
	input_ = np.array([this_np_array[-30:]])
	input_ = np.reshape(input_, (input_.shape[0], input_.shape[1], 1))
	new_input = input_


	#Recursively Predict Next 7 days prices
	output_ = model.predict(input_, min_price, max_price, mean_price)
	outputs = [de_normalize(output_, min_price, max_price, mean_price)]
	for i in range(0, 29):
		new_input = np.append(new_input[0][-29:], output_)
		new_input = np.reshape(np.array(new_input), (1, 30, 1))
		# print(new_input)
		print(new_input.shape)
		output_ = model.predict(new_input, min_price, max_price, mean_price)
		outputs.append(de_normalize(output_, min_price, max_price, mean_price))

	print(outputs)

	x = list(range(1, 31))

	fig, axs = plt.subplots(2)
	fig.suptitle(name)

	testing = pd.read_csv("./training_files/testing.csv")
	new = testing.loc[testing['ID'] == int(ID)]["price"]
	new = np.array(new.astype(int))[-30:]

	axs[0].plot(x, outputs)
	axs[0].title.set_text('Prediction')
	axs[1].plot(x, new, label = 'Actual')
	axs[1].title.set_text('Actual')

	plt.savefig("./predictions/"+ name + ".png")

def de_normalize(output_, min_price, max_price, mean_price):
	return output_[0][0] * (max_price - min_price) + mean_price

def min_max_normalize(df, min_price, max_price, mean_price):
	return (df - mean_price) / (max_price - min_price)

if __name__ == '__main__':
	f = open('./utils/fodders.json')
	fodder_dict = json.load(f)

	# Sergio Busquets
	for key, value in fodder_dict.items():
		lstm_predict(value, key)
