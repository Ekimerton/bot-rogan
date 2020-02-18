import tweepy
import os
from ai.predict import generate_tweet

consumer_token = os.environ['consumer_token']
consumer_token_secret = os.environ['consumer_token_secret']
access_token = os.environ['access_token']
access_token_secret = os.environ['access_token_secret']

auth =tweepy.OAuthHandler(consumer_token, consumer_token_secret)
auth.set_access_token(access_token, access_token_secret)
tp = tweepy.API(auth)

tweet = generate_tweet("old", 200, 0.6)
tp.update_status(tweet)
