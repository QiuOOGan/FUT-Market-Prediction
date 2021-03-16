# FUT-Market-Prediction
In this project, we will try to use train a LSTM to make predictions on the fodders on FUT Market.

## Fodders:
* Fodders are those high rated but not very useful hence relatively cheap cards.
* When EA Release a good SBC, such as POTM Lionel Messi, the prices of fodders will fluctuate.

## Crawl:
* There are 207 such fodders, we have saved them as time series csv files.

#### To get most recent 207 fodders time series price, run:
```sh
python3 get_player_TimeSeries_Fodder_Only.py
```
