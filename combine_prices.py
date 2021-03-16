import json
import os
import glob
import pandas as pd
import numpy as np


def combine_all_fodders():
    directory = os.path.join("./FIFA21_Players_TimeSeries/")
    all_files = glob.glob(directory + "*")

    big_df_all = pd.DataFrame()
    big_df_training = pd.DataFrame()
    big_df_testing = pd.DataFrame()
    for file in all_files:
        this_df = pd.read_csv(file)
        length = len(this_df)

        # skip first week when there might be no supplies
        big_df_training = big_df_training.append(this_df[7:length-45], ignore_index=True)

        # testing set is the last 45 days
        big_df_testing = big_df_testing.append(this_df[length-45:], ignore_index=True)

        # save all data for final training
        big_df_all = big_df_all.append(this_df, ignore_index=True)


    big_df_all.to_csv("./training_files/all.csv")
    big_df_training.to_csv("./training_files/training.csv")
    big_df_testing.to_csv("./training_files/testing.csv")
    print("Done, combined all fodders")

if __name__ == '__main__':
    combine_all_fodders()