import os
import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc
import emoji
import re
import tqdm
from multiprocessing import cpu_count, Pool
import numpy as np


def fetch_data():
    """
    fetch the data from the github

    """
    url = "https://raw.githubusercontent.com/chouhbik/Sentiment-Analysis-of-Tweets/master/ExtractedTweets.csv"
    data = pd.read_csv(url)
    data["target"] = data["Party"].apply(lambda x: 0 if x == "Democrat" else 1)

    return data


def preprocess_text(text):
    """
    clen a string of words and return a list of lemmatized tokens
    """

    nlp = spacy.load("en_core_web_sm")

    # Lowercase the text
    text = text.lower()
    # Remove sentences starting with 'RT ' or '@' (hashtags)
    if text.startswith("rt "):  # or text.startswith("@"):
        return None  # Empty list for removal
    # Remove whitespace
    text = text.strip()
    # Remove numbers
    text = re.sub(r"[0-9]+", "", text)
    # Remove URLs
    text = re.sub(r"(http|https)?://[^\s]+", "", text)  # Match and replace URLs
    text = re.sub(r"@\w+", "", text)
    # Remove emojis using emoji library
    text = emoji.demojize(text)  # Replaces emojis with their corresponding description
    # Tokenize the text
    doc = nlp(text)
    tokens = [token for token in doc]
    # Remove stopwords
    # tokens = [token for token in tokens if not token.is_stop or not token.is_punct]
    cleanded_tokens = []
    for each in tokens:
        if each.is_punct or each.is_stop:
            continue
        else:
            cleanded_tokens.append(each)

    # Lemmatize the tokens
    lemmas = [token.lemma_ for token in cleanded_tokens if token.text != " "]

    return lemmas


def preprocess_tweet(tweet):
    return preprocess_text(tweet)  # REPLACE WITH UR FUNCTION


def parallelize_dataframe(df, func, n_cores=cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(list(tqdm(pool.imap(func, df_split), total=len(df_split))))
    pool.close()
    pool.join()
    return df


# Apply the preprocessing to a DataFrame chunk
def apply_preprocessing(df_chunk):
    df_chunk["tweet_clean"] = df_chunk["tweet"].apply(preprocess_tweet)
    return df_chunk


if __name__ == "__main__":

    data = fetch_data().sample(frac=0.05)
    df_sample = parallelize_dataframe(data, apply_preprocessing)
    df_sample.to_csv("data_cleaned.csv", index=False)

    # full data cleaning
    # for person in data_dict.keys():
    #     for tweets in tqdm.tqdm(data_dict[person]):
    #         X_tweet.append(preprocess_text(tweets))
    #         X_target.append(person[1])
