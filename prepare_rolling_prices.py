import json
import os
import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing

def prepare_rolling_prices(df):
	f = open('./utils/fodders.json')
	fodder_dict = json.load(f)
	global min_price
	global max_price
	global mean_price
	min_price = min(pd.array(df["price"]))
	max_price = max(pd.array(df["price"]))
	mean_price = np.mean(pd.array(df["price"]))

	rolling_x = []
	rolling_y = []

	for key, value in fodder_dict.items():
		this_df = df.loc[df['ID'] == int(value)]["price"]
		this_df = this_df.astype(int)
		
		# min max normalization
		this_df = min_max_normalize(this_df)
		this_np_array = np.array(this_df)
		# Create Rolling Prices

		for i in range(30, len(this_np_array)):
			this_date = []
			for j in range(i-30, i):
				this_date.append(this_np_array[j])
			rolling_x.append(this_date)
			rolling_y.append(this_np_array[i])


	return pd.DataFrame(rolling_x), pd.DataFrame(rolling_y) 


def min_max_normalize(df):
	return (df - mean_price) / (max_price - min_price)

if __name__ == '__main__':
	training = pd.read_csv("./training_files/training.csv")
	x, y = prepare_rolling_prices(training)
	print(x, y)
