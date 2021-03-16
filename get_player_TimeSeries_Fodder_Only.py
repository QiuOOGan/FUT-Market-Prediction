import json
import os
import glob
import pandas as pd
import numpy as np
from get_player_TimeSeries import is_special, append_rows, get_player_file

def generate_fodders():
	fodder_dict = {}
	directory = os.path.join("./FIFA21_Players_TimeSeries/")
	all_files = glob.glob(directory + "*")
	isHeader = True
	for file in all_files:
		temp = pd.read_csv(file)
		name = temp["name"][0]
		ID = temp["ID"][0]
		if not is_fodder(name):
			print(name, " is not a fodder!")
			continue
		fodder_dict[name] = str(ID)

	print(fodder_dict)
	with open('./utils/fodders.json', 'w') as fp:
		json.dump(fodder_dict, fp, sort_keys=True, indent=4)

def is_fodder(name):
	norm_name = name.lower()
	suses = ["messi", "ronaldo", "neymar", "mbapp"]
	return not any(sus in norm_name for sus in suses)

def get_player_TimeSeries_Fodder_Only():
	f = open('./utils/fodders.json')
	fodder_dict = json.load(f)

	# start scraping fodder only
	for name in fodder_dict:
		ID = fodder_dict[name]

		data_list = get_player_file(ID)
		if not data_list: continue

		df = pd.DataFrame(data_list)
		df.columns = ["futbin_id", "ID", "name", "rating", "price", "date"]
		file_name = "./FIFA21_Players_TimeSeries/" + df['name'][0]+".csv"
		df.to_csv(file_name)
		print("successfully scraped ", df['name'][0], "\n")
		
	print("Done")

# Only scrape fodders: only 207 players (removed Messi, Ronaldo, Mbappe and Neymar)
# TODO: Maybe remove more? like Rashford. Or can remove outliers based on their prices
if __name__ == '__main__':
	# generate_fodders()
	get_player_TimeSeries_Fodder_Only()
