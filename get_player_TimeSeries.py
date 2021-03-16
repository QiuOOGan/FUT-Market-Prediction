import pandas as pd
import numpy as np
import requests
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import re

domain = 'https://www.futbin.com'
version = 21
section = 'player'
specials = ["icon", "inform", "champions", "potm", "toty", "tott", "ones to watch"]


def is_special(title):
	norm_title = title.lower()
	return any(special in norm_title for special in specials)

def append_rows(name, futbin_id, rating, ID):
	r = requests.get('https://www.futbin.com/21/playerGraph?type=daily_graph&year=21&player={0}'.format(futbin_id))
	data = r.json()

	#Change ps to xbox or pc to get other prices
	data_list = []
	for price in data['ps']:
	#There is extra zeroes in response.
		date = datetime.utcfromtimestamp(price[0] / 1000).strftime('%Y-%m-%d')
		price = price[1]
		# print(date,price)
		data_list.append([futbin_id, ID, name, rating, price, date])
	return data_list

def get_player_file(ID):
	# for ID in range(min, max):
	url = "%s/%s/%s/%s"% (domain, version, section, ID)
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	obj = soup.find("div", {"id": "page-info"})
	if not obj:
		print("ID: ", ID, " does not exist")
		return None

	FUTBIN_ID = obj['data-player-resource']
	print("Counter_ID", ID)
	print("FUTBIN_ID: ", FUTBIN_ID)

	# Rating
	TITLE = str(soup.find('title').string)
	keyword = "FIFA 21 - "
	ind2 = TITLE.find(keyword)
	start = ind2+len(keyword)

	FULL_NAME = TITLE[:ind2].replace(" ", "")

	# Check if special version, if, skip
	# if special it's not fodder
	if is_special(TITLE):
		print("skip ", FULL_NAME, " is not a fodder due to version\n")
		return None

	# Check if rating reasonable
	RATING = int(TITLE[start:start + 2])
	if RATING < 83 or RATING > 93:
		print("skip ", FULL_NAME, " is not a fodder due to rating: ", RATING, "\n")
		return None

	return append_rows(FULL_NAME, FUTBIN_ID, RATING, ID)

def scrape(start):
	for i in range(start, start+10000):
		data_list = get_player_file(i)

		with open("./utils/last_ID.txt", "w") as f:
			f.write(str(i))

		if not data_list: continue

		df = pd.DataFrame(data_list)
		df.columns = ["futbin_id", "ID", "name", "rating", "price", "date"]
		file_name = "./FIFA21_Players_TimeSeries/" + df['name'][0]+".csv"
		df.to_csv(file_name)
		print("successfully scraped ", df['name'][0], "\n")

if __name__ == '__main__':
	with open("./utils/last_ID.txt", "r") as f:
	    last = int(f.read())
	    scrape(last+1)

