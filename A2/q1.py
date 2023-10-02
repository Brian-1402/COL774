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


def generate_vocab(sentences):
    vocab = dict()
    for sentence in sentences:
        for word in sentence:
            vocab[word] = 1
    return vocab


# v = generate_vocab(corona_train_df["CoronaTweet"])


class NaiveBayes:
    def __init__(
        self,
        class_mapping={"Negative": 0, "Neutral": 1, "Positive": 2},
        class_header="Sentiment",
        data_header="CoronaTweet",
    ):
        self.y_id = class_mapping
        self.k = len(self.y_id)
        self.phi_x = []
        for i in range(self.k):
            self.phi_x.append(dict())
        self.phi_y = [0] * self.k
        self.class_header = class_header
        self.data_header = data_header

        self.y_id_inv = dict()
        for key in self.y_id.keys():
            value = self.y_id[key]
            self.y_id_inv[value] = key
            # For using in inference

    def train_params(self, df):
        k = self.k
        phi_x = self.phi_x
        phi_y = self.phi_y
        phi_x_denom = [0] * k

        # O(m) or O(words in data)
        for r in range(len(df)):
            y = self.y_id[df.loc[r, self.class_header]]
            sentence = df.loc[r, self.data_header]
            for word in sentence:
                d = phi_x[y]
                # If word hasn't been encountered before, intantiate it's value as 0 across all the phi_x dicts.
                # Benefit of doing this, we won't get errors when a word exists in one dict but not in another.
                # Another benefit is that every phi_x can be used as a global vocab.
                # Drawback might be that slightly more memory usage.
                if word not in d:
                    for yi in range(k):
                        phi_x[yi][word] = 0
                d[word] += 1
                phi_x_denom[y] += 1
            phi_y[y] += 1

        V = len(phi_x[0])
        m = df.shape[0]

        # O(V)
        # dividing by denom and Laplace smoothing
        for word in phi_x[0].keys():
            for yi in range(k):
                phi_x[yi][word] += 1
                phi_x[yi][word] /= phi_x_denom[yi] + V
        for yi in range(k):
            phi_y[yi] /= m

    def inference(self, sentence):
        V = len(self.phi_x[0])
        lpy = [0] * self.k
        # for r in range(len(series)):
        # sentence = series.loc(r, "CoronaTweet")
        for yi in range(self.k):
            lpx_y = 0
            for word in sentence:
                pxi_y = self.phi_x[yi][word] if word in self.phi_x[0] else 1 / V
                lpx_y += np.log10(pxi_y)  # P(X=sentence|Y) = Prod[ P(X=word|Y) ]
            lpy[yi] = lpx_y + np.log10(self.phi_y[yi])
            # P(Y=yi|X=x) ~ P(X=x|Y=yi) * P(Y=yi)
        return self.y_id_inv[lpy.index(max(lpy))]

    def test(self, df, inf_func=None):
        inf_func = inf_func if inf_func else self.inference
        cm = np.zeros((self.k, self.k))
        for r in range(len(df)):
            sentence = df.loc[r, self.data_header]
            inf = self.y_id[inf_func(sentence)]
            real = self.y_id[df.loc[r, self.class_header]]
            cm[real, inf] += 1
        return cm.trace() / cm.sum(), cm


nb = NaiveBayes()
nb.train_params(corona_train_df)
test_set_result = nb.test(corona_validation_df)
train_set_result = nb.test(corona_train_df)

print("Accuracy on test", test_set_result[0], "\n", test_set_result[1])
print("Accuracy on train", train_set_result[0], "\n", train_set_result[1])
