from tqdm import tqdm
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import json
import re
import sys
import nltk
from nltk.corpus import stopwords


def preprocess():
    """
    create a dataframe with 'Tweet', 'gpt_class', 'clean_tweet'
    """

    def get_class(tweet):
        """
        Label a tweet using OpenAI API:
        Take a single tweet as input and API will return one of the following label 'Democrat', 'Republican', 'Neutral'.
        API could also return None if LLM is not able to tag the input
        """

        # Set your OpenAI API key
        client = OpenAI(api_key=api_key)

        try:
            prompt = """You will be given a set of Twitter posts from different US politicians. Your task is to use your knowledge of
            US politics to make an educated guess on whether the
            poster is a Democrat or Republican. Respond either ‘Democrat’ or ‘Republican’.
            If the message does not have enough information for an educated guess respond with 'Neutral'."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Classify this tweet: {tweet}"},
                ],
                functions=[
                    {
                        "name": "classify_tweet",
                        "description": "Function will classify tweet into one of this classes: Democrat/Republican/Neutral",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "class": {
                                    "type": "string",
                                    "description": "One of this classes: Democrat/Republican/Neutral",
                                }
                            },
                            "required": ["class"],
                        },
                    }
                ],
            )
            prediction = response.choices[0].message.function_call.arguments
            prediction = json.loads(prediction)["class"]

            return prediction

        except:

            return None

    def label_tweets(df):
        """
        Take a dataframe where a column named 'Tweet' is present, write to csv the same dataframe with an additional column called
        'gpt_class' with the label created by get_class function

        """
        for index, row in tqdm(df.iterrows()):
            row["gpt_class"] = get_class(row["Tweet"])
            row.to_frame().T.to_csv(
                "tweets_gtp_tagged.csv",
                mode="w" if index == 0 else "a",
                header=True if index == 0 else False,
                index=False,
            )
        print("tweets_gpt_tagged.csv created ✅")

    def process_hashtag(hashtag):
        """
        function to process hashtags

        """
        hashtag_body = hashtag[1:]
        if hashtag_body.upper() == hashtag_body:
            return f"<HASHTAG> {hashtag_body} <ALLCAPS>"
        else:
            parts = re.split(r"(?=[A-Z])", hashtag_body)
            return " ".join(["<HASHTAG>"] + parts)

    def process_tweet(input):
        input = input.lower()
        # remove the short urls first
        input = re.sub(r"https?:\/\/t\.co\/[A-Za-z0-9]+", "", input)
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"
        input = input.replace("&amp", " ")
        pattern = r"[^\w\s]"
        input = re.sub(pattern, " ", input)
        input = input.replace("\n", " ")
        # pattern = r"<.*?>"
        # input = re.sub(pattern, "", input)
        input = input.replace(
            "/", " / "
        )  # Force splitting words appended with slashes (once we tokenized the URLs, of course)
        input = re.sub(r"@\w+", "", input)
        input = re.sub(
            rf"{eyes}{nose}[)d]+|[)d]+{nose}{eyes}", "<SMILE> ", input, flags=re.I
        )
        input = re.sub(rf"{eyes}{nose}p+", "<LOLFACE>", input, flags=re.I)
        input = re.sub(rf"{eyes}{nose}\(+|\)+{nose}{eyes}", "<SADFACE> ", input)
        input = re.sub(rf"{eyes}{nose}[\/|l*]", "<NEUTRALFACE> ", input)
        input = re.sub(r"<3", "<HEART> ", input)
        input = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER> ", input)

        input = re.sub(r"#\S+", lambda hashtag: process_hashtag(hashtag.group()), input)
        input = re.sub(
            r"([!?.]){2,}", lambda match: f"{match.group(0)[0]} <REPEAT> ", input
        )
        input = re.sub(
            r"\b(\S*?)(.)\2{2,}\b",
            lambda match: f"{match.group(1)}{match.group(2)} <ELONG> ",
            input,
        )
        input = re.sub(
            r"([^a-z0-9()<>\'`\-]){2,}", lambda word: f"{word.group(0).lower()}", input
        )
        input = " ".join([word.strip() for word in input.split()])

        # remove stopwords that are not negation
        filtered_words = [
            word
            for word in input.split()
            if word.lower() not in stop_words_without_negations
        ]
        input = " ".join(filtered_words)

        return input

    breakpoint()
    path = input(
        "Input the path of the raw dataframe, a column called tweet must be present: "
    )
    data = pd.read_csv(path)
    # take only the Tweet column from the input dataframe
    data = data[["Tweet"]]

    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    api_key = input("Insert OpenAI API key: ")
    label_tweets(data)

    data_labeled = pd.read_csv("tweets_gtp_tagged.csv").dropna()

    # create a list of stop words without neagtionns
    stop_words = set(stopwords.words("english"))
    # Define a list of common negations words
    negations = {
        "no",
        "not",
        "nor",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "never",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "doesn't",
        "don't",
        "didn't",
        "won't",
        "wouldn't",
        "shan't",
        "shouldn't",
        "can't",
        "cannot",
        "couldn't",
        "mustn't",
        "let's",
        "mightn't",
        "needn't",
    }
    # Filter stopwords to exclude negations
    stop_words_without_negations = stop_words - negations

    data_labeled["clean_tweet"] = data_labeled["Tweet"].apply(process_tweet)

    data_labeled.to_csv("data_labeled_cleaned.csv")

    print("data_labeled_cleaned.csv created ✅")


if __name__ == "__main__":
    preprocess()
