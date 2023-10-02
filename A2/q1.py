import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


corona_train_df = pd.read_csv(abs_path("data/nb/Corona_train.csv"), header=0)
corona_validation_df = pd.read_csv(abs_path("data/nb/Corona_validation.csv"), header=0)

# print(Corona_train_df["CoronaTweet"].dtype)


def tokenize_main(s):
    """First cleans the sentences and tokenizes"""
    # tokenized = tokenizer.tokenize(s)
    tokenized = s.split()
    # print(tokenized)
    return tokenized


def preprocessing(s):
    return tokenize_main(s)


def clean_tweet_data(df):
    tweet_ser = df["CoronaTweet"]
    tweet_ser_tokenized = tweet_ser.apply(preprocessing)
    df.drop(["CoronaTweet"], axis=1, inplace=True)
    df.insert(2, "CoronaTweet", tweet_ser_tokenized)


clean_tweet_data(corona_train_df)
clean_tweet_data(corona_validation_df)
