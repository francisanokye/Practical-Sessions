import tweepy
from textblob import TextBlob

consumer_key = 'vn6Qmp3kRx4TlIQhFwdmG4Jg3'
consumer_key_secret = '3tx275hm3ZW7YdMLgZzN5RrBN0biibruOQnJlLw3aexYDeVGc5'

access_token = '815175746612174848-0mGtAqfAYu87ziDy6NcvT3HH48Yvzus'
access_token_secret = 'ar1lGNaHiUiGebV0vvNqy4yh3JYhtKLEzUwoNxXsEejKK'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Penguins')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
	if analysis.sentiment[0]>0:
		print ('Positive')
	else:
		print ('Negative')
	print("")
