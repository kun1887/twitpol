import pandas as pd
import spacy
import emoji
import re
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def fetch_data():
    """
    Fetch the data from the GitHub repository
    """
    url = "https://raw.githubusercontent.com/chouhbik/Sentiment-Analysis-of-Tweets/master/ExtractedTweets.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower()
    df["y"] = df.party.apply(lambda x: x == "Democrat")
    df = df.drop(["party", "handle"], axis=1)
    return df


# Load SpaCy model and add emoji component
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """
    Clean a string of words and return a list of lemmatized tokens
    """
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
    # Remove stopwords and punctuation
    cleaned_tokens = [
        token for token in tokens if not token.is_punct and not token.is_stop
    ]
    # Lemmatize the tokens
    lemmas = [token.lemma_ for token in cleaned_tokens if token.text != " "]
    return lemmas


def preprocess_tweet(tweet):
    return preprocess_text(tweet)


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
    data = fetch_data().sample(frac=0.3)
    data = data.loc[~data.tweet.str.contains("RT ")]
    df_sample = parallelize_dataframe(data, apply_preprocessing)
    df_sample.to_csv("data_cleaned.csv", index=False)
