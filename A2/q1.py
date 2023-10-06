import os, re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


corona_train_df = pd.read_csv(abs_path("data/nb/Corona_train.csv"), header=0)
corona_validation_df = pd.read_csv(abs_path("data/nb/Corona_validation.csv"), header=0)

# print(Corona_train_df["CoronaTweet"].dtype)


class NaiveBayes:
    def __init__(
        self,
        class_mapping={"Negative": 0, "Neutral": 1, "Positive": 2},
        class_header="Sentiment",
        data_header="CoronaTweet",
        bigram=False,
        bigram_weight=0.85,
    ):
        self.y_id = class_mapping
        self.k = len(self.y_id)
        self.phi_x = [dict() for i in range(self.k)]
        self.phi_y = [0] * self.k
        self.class_header = class_header
        self.data_header = data_header
        self.logV = 0
        self.bigram = bigram
        self.bigram_weight = bigram_weight

        self.phi_x2 = [dict() for i in range(self.k)]

    def train_params(self, df):
        k, phi_x, phi_y = self.k, self.phi_x, self.phi_y
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
        for yi in range(k):
            for word in phi_x[yi]:
                phi_x[yi][word] += 1
                phi_x[yi][word] /= phi_x_denom[yi] + V
            phi_y[yi] = np.log10(phi_y[yi]) - np.log10(m)
        # for yi in range(k):
        self.logV = np.log10(len(phi_x[0]))

        if self.bigram:
            self.train_bigrams(df)

    def train_bigrams(self, df):
        k, phi_x2 = self.k, self.phi_x2
        phi_x2_denom = [0] * self.k
        for r in range(len(df)):
            y = self.y_id[df.loc[r, self.class_header]]
            sentence = df.loc[r, self.data_header]
            for i in range(len(sentence)):
                word = sentence[i]
                prev = "<start>" if i == 0 else sentence[i - 1]
                d = phi_x2[y]
                if prev not in d:
                    for yi in range(k):
                        phi_x2[yi][prev] = {word: 0}
                elif word not in d[prev]:
                    for yi in range(k):
                        phi_x2[yi][prev][word] = 0
                d[prev][word] += 1
                phi_x2_denom[y] += 1
            # phi_y[y] += 1
        V = len(self.phi_x[0])
        for yi in range(k):
            for prev in phi_x2[yi]:
                for word in phi_x2[yi][prev]:
                    phi_x2[yi][prev][word] += 1
                    phi_x2[yi][prev][word] /= phi_x2_denom[yi] + V

            # phi_y[yi] = np.log10(phi_y[yi]) - np.log10(m)

        # for prev in phi

    def inference(self, sentence):
        V = len(self.phi_x[0])
        lpy = [0] * self.k
        # for r in range(len(series)):
        # sentence = series.loc(r, "CoronaTweet")
        for yi in range(self.k):
            lpx_y = 0
            for i in range(len(sentence)):
                word = sentence[i]
                pxi_y = 0
                if not self.bigram:
                    pxi_y = (
                        np.log10(self.phi_x[yi][word])
                        if word in self.phi_x[yi]
                        else -self.logV
                    )
                else:
                    prev = "<start>" if i == 0 else sentence[i - 1]
                    uni = self.phi_x[yi][word] if word in self.phi_x[yi] else 1 / V
                    bi = 1 / V
                    if prev in self.phi_x2[yi]:
                        if word in self.phi_x2[yi][prev]:
                            bi = self.phi_x2[yi][prev][word]
                    w = self.bigram_weight
                    pxi_y = np.log10((1 - w) * uni + w * bi)

                lpx_y += pxi_y  # P(X=sentence|Y) = Prod[ P(X=word|Y) ]
            lpy[yi] = lpx_y + self.phi_y[yi]
            # P(Y=yi|X=x) ~ P(X=x|Y=yi) * P(Y=yi)
        return ["Negative", "Neutral", "Positive"][np.argmax(lpy)]

    def test(self, df, inf_func=None):
        inf_func = inf_func if inf_func else self.inference
        cm = np.zeros((self.k, self.k))
        for r in range(len(df)):
            sentence = df.loc[r, self.data_header]
            inf = self.y_id[inf_func(sentence)]
            real = self.y_id[df.loc[r, self.class_header]]
            cm[real, inf] += 1
        return cm.trace() / cm.sum(), cm

    def print_test_result(self, train_df, test_df, inf_func=None, print_matrix=False):
        train_val = self.test(train_df, inf_func)
        test_val = self.test(test_df, inf_func)
        print(f"Accuracy on training data: {100*train_val[0]:.2f} %")
        if print_matrix:
            print(
                "\n\nConfusion matrix:\n",
                train_val[1],
                f"\nTrace: {train_val[1].trace()}\n",
                "\n",
            )
        print(f"Accuracy on validation data: {100*test_val[0]:.2f} %")
        if print_matrix:
            print(
                "\n\nConfusion matrix:\n",
                test_val[1],
                f"\nTrace: {test_val[1].trace()}\n",
            )


"""Data Preproccessing"""


import html, nltk, wordsegment
import wordcloud

# wordsegment.load()

custom_stopwords = {"https", "t", "co", "amp"}
nltk_stopwords = set(nltk.corpus.stopwords.words("english"))
# nltk.download("stopwords")
# nltk.download("wordnet")
stopwords = nltk_stopwords.union(custom_stopwords)
# tw_tokenizer = nltk.tokenize.TweetTokenizer()
re_tokenizer = nltk.tokenize.RegexpTokenizer("[a-z0-9]+'*[a-z0-9]*")
# re_tokenizer only outputs alphanumeric with ' allowed in between chars(they'll, etc), no #, @ or other symbols.

stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocessing_basic(s):
    return s.split()


def segment_hashtags(token):
    """Deals with hashtags and -/_ containing words by separating joint words and lemmatizing them individually.
    segment also cleans symbols.

    But takes too much time, with no improvement in accuracy.
    """
    ans = []
    if re.search(r"^#|[-_]", token):
        seg = wordsegment.segment(token)
        for word in seg:
            if len(token) > 1:
                ans.append(lemmatizer.lemmatize(word))
    return ans


def process_tokens(s_tokens):
    ans = []
    for token in s_tokens:
        if token in stopwords or token.isnumeric() or len(token) < 3:
            continue
        ans.append(lemmatizer.lemmatize(token))
        # ans.append(stemmer.stem(token))
    return ans


def process_sentence(s):
    s = html.unescape(s)  # convert &amp; to &, &lt; to < etc.
    s = s.lower()
    # s = unicodedata.normalize("NFKD", s)
    # # converts non ascii characters to their closest ascii char. Like ā to a.
    s = s.encode("ascii", "ignore").decode("utf-8", "ignore")
    # removes non-ascii characters
    s = re.sub(r"https?\S*|t.co\S*|@\S*", "", s, count=0)
    # remove urls and @tags

    s_tokens = re_tokenizer.tokenize(s)
    processed = process_tokens(s_tokens)
    return processed


def preprocess_tweets(df, preprocessing_func):
    tweet_ser = df["CoronaTweet"]
    tweet_ser_tokenized = tweet_ser.apply(preprocessing_func)
    df_new = df.drop(["CoronaTweet"], axis=1)
    df_new.insert(2, "CoronaTweet", tweet_ser_tokenized)
    return df_new


"""Word clouds"""


def generate_word_cloud(data):
    stopwords = set(wordcloud.STOPWORDS).union(custom_stopwords)
    wc = wordcloud.WordCloud(
        background_color="white",
        max_words=2000,
        stopwords=stopwords,
        collocations=False,
    )
    data = " ".join(data)
    wc.generate(data)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


"""Misc"""


def df_to_list(df, data_header="CoronaTweet"):
    data = df[data_header].explode().tolist()
    data = [str(i) for i in data]
    return data


"""Main execution"""


pre_time = datetime.now()
train_df = preprocess_tweets(corona_train_df, process_sentence)
validation_df = preprocess_tweets(corona_validation_df, process_sentence)
pre_time = datetime.now() - pre_time
print("preprocessing time:", pre_time.total_seconds(), "sec\n")

# train_df.to_csv(abs_path("data/nb/clean.csv"))
generate_word_cloud(df_to_list(train_df))
# generate_word_cloud(df_to_list(validation_df))

nb = NaiveBayes(bigram=True)

train_time = datetime.now()
nb.train_params(train_df)
train_time = datetime.now() - train_time
print("Training time:", train_time.total_seconds(), "sec\n")

inf_time = datetime.now()
nb.print_test_result(train_df, validation_df)
inf_time = datetime.now() - inf_time
print("Inference time:", inf_time.total_seconds(), "sec")


def part_b_c():
    def random_model(x):
        return ["Negative", "Neutral", "Positive"][np.random.randint(0, 3)]

    def const_model(x, i=2):
        return ["Negative", "Neutral", "Positive"][i]

    print("\nFor model:\n")
    nb.print_test_result(train_df, validation_df)
    print("\nFor random:\n")
    nb.print_test_result(train_df, validation_df, random_model)
    print("\nFor positive model:\n")
    nb.print_test_result(train_df, validation_df, const_model)
    print()


# part_b_c()
