import streamlit as st
import tweepy
import pandas as pd
import matplotlib.pyplot as plt

# Twitter API credentials
#API_KEY = st.secrets("API_KEY")
API_KEY=""
API_SCRET_KEY = ''
ACESS_TOKEN = ''
ACESS_TOKEN_SECRET = ''

#autenticate with tt API
auth = tweepy.OAuthHandler(API_KEY, API_SCRET_KEY)
auth.set_access_token(ACESS_TOKEN, ACESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Load pre-trained model
model = ''
# function to fetch text from url
def get_tweet_text(url):
    tweet_id = url.split('/')[-1]
    tweet = api.get_status(tweet_id, tweet_mode='extended')
    return tweet.full_text

# function to classify text
def classify_text(text):
    pred = model.predict([text])
    return pred[0]

# initialize storage for historical data
if 'history' not in st.session_state:
    st.session_state.history = []

# streamlit app
st.title("Twitpol")

option = st.selectbox("Choose input type", ("TWITTER URL", "Text"))

## to insert  Twitter url

if option == "TWITTER URL":
    url = st.text_input("Enter Twitter Url")
    if st.button("Classify"):
        try:
            text = get_tweet_text(url)
            label = classify_text(text)
            st.write(f"The tweet is classified as: {label}")
            st.session_state.history.append({'text': text, 'label':label})
        except Exception as e:
            st.write("Error fetching the tweet: ", e)

## to insert the text from a tweet

else:
    text = st.text_area("Enter Text")
    if st.button("Classify"):
        label = classify_text(text)
        st.write(f"The text is classified as: {label}")
        st.session_state.history.append({'text': text, 'label': label})


# display historical data
if st.session_state.history:
    st.write("### Historical Data")
    history_df = pd.DataFrame(st.session_state.history)
    st.write(history_df)

# plot historical data
    st.write("### Historical Data Plot")
    history_count = history_df['label'].value_counts()
    fig, ax = plt.subplots()
    history_count.plot(kind='bar', ax=ax)
    st.pyplot(fig)
