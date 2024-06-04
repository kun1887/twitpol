import os
import pandas as pd
from pathlib import Path
import spacy
from spacy.tokens import Doc
import emoji
import re


def fetch_data():
    """
    fetch the data from the csv files stored in /Users/kun/code/kun1887/TwitPol/raw_data
    return a dictionary with key = (name, target) and value = dataframe with one tweet coloum

    """

    files = os.listdir("/Users/kun/code/kun1887/TwitPol/raw_data")
    data = {}

    for file in files:

        df = pd.read_csv(f"/Users/kun/code/kun1887/TwitPol/raw_data/{file}")
        target = file[0]
        tweets = list(df.Tweet.values)
        key = (file.split("_")[0][1:], target)
        data.update({key: tweets})

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


import tqdm

if __name__ == "__main__":

    data_dict = fetch_data()

    X_cleaned = pd.DataFrame(columns=["tweet", "target"])
    X_tweet = []
    X_target = []

    for person in data_dict.keys():
        for tweets in tqdm.tqdm(data_dict[person]):
            X_tweet.append(preprocess_text(tweets))
            X_target.append(person[1])

    X_cleaned["tweet"] = X_tweet
    X_cleaned["target"] = X_target

    print(X_cleaned)
    X_cleaned.to_csv("Cleaned_data.csv", sep=",", index=False, encoding="utf-8")
