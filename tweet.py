import tweepy
import os
from ai.predict import generate_tweet

CO

auth = tweepy.OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")

tweet = generate_tweet("old", 200, 0.7)
print(tweet)
