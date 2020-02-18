import tweepy
import os
from ai.predict import generate_tweet
import time

interval = 60 * 60 * 12 # 12 hours
#interval = 15

consumer_token = os.environ['consumer_token']
consumer_token_secret = os.environ['consumer_token_secret']
access_token = os.environ['access_token']
access_token_secret = os.environ['access_token_secret']

auth = tweepy.OAuthHandler(consumer_token, consumer_token_secret)
auth.set_access_token(access_token, access_token_secret)
tp = tweepy.API(auth)

while True:
    tweet = generate_tweet("", 140, 1)
    tp.update_status(tweet)
    time.sleep(interval)

