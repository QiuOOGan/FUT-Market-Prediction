# FUT-Market-Prediction
In this project, we will try to train a LSTM to make predictions on the fodders on FUT Market.

## Fodders:
* Fodders are those high rated but not very useful hence relatively cheap cards.
* When EA Release a good SBC, such as POTM Lionel Messi, the prices of fodders will fluctuate greatly.
  * Specifically, we mostly scrape gold rare players that are above 83 rating (inclusive).
* Players such as Cristiano Ronnaldo, Lionel Messi, Neymar, etc are not considered fodders, since literally nobody will buy them on the transfer market to put them into an SBC.
* Also, special cards like Icon are removed as well for obvious reason.

## Crawl:
* There are 207 such fodders, we have saved them as day-level time series csv files. 
* To get most recent 207 fodders time series price, run:
  ```sh
  python3 get_player_TimeSeries_Fodder_Only.py
  ```
* We will potentially scrape hour-level time series data in the future.
