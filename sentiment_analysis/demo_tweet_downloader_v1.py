#This is a script to download tweets and perform some analysis on the data.
#This script was written by Vincent Appiah
#Email:appiahv@rocketmail.com
#The consumerkey and accesstoken is valid from April 25 - 27 2019

consumerkey=''
consumersecret=''

accesstoken=''
accesstokensecret=''



import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re

"""
try changing the user accounts with any of the following
 '''nm_appiah
   realDonalTrump
   ghdatascience
   shattawalegh
   stonebwoyb
   '''
"""   
# funtion to authenticate to the twitter platform
def authenticate(consumerkey,consumersecret,accesstoken,accesstokensecret):
    auth=tweepy.OAuthHandler(consumerkey,consumersecret)
    auth.set_access_token(accesstoken,accesstokensecret)
    return auth

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())

def analyze_sentiment(tweet):
    analysis=TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity>0:
        return(1)
    elif analysis.sentiment.polarity<0:
        return(-1)
    else:
        return(0)
def visualize_likes(df):
    time_likes=pd.Series(data=df['likes'].values,index=df['date'])
    time_likes.plot(figsize=(16,4),color='b',label='likes')
    plt.show()

def visualize_all(df,args):
    for i in args:
        time_likes=pd.Series(data=df[i].values,index=df['date'])
        time_likes.plot(figsize=(16,4),label=i,legend=True)
    plt.show()
def tweets_stats(df):   # a little stats
    
    # average length of tweets
    avg=df['length'].mean()
   

    # maximum number of likes
    maxlikes=df['likes'].max()

    #maximum number of retweets
    maxretweetcount=df['retweets'].max()


    print('Average tweet : %s' % int(avg))
    print('maximum likes :%s'%maxlikes)
    print('maximum retweets :%s'%maxretweetcount)

def tweets_to_df(tweet):     # function to save tweets to a dataframe
    global df
    df =pd.DataFrame()
    df['id']=[tweet.id for tweet in tweets]
    df['tweet']=[tweet.text for tweet in tweets]
    df['date']=[tweet.created_at for tweet in tweets]
    df['source']=[tweet.source for tweet in tweets]
    df['likes']=[tweet.favorite_count for tweet in tweets]
    df['retweets']=[tweet.retweet_count for tweet in tweets]
    df['length']=[len(tweet.text) for tweet in tweets]

    #sentiment analysis . you can uncomment the line add the sentiment
    df['sentiment']=[analyze_sentiment(tweet.text) for tweet in tweets]
    
    return(df)

def visualize_tweet_source(df):
    #frequency of tweet sources
    source_counts=df['source'].value_counts()
    labels=source_counts.index
    counts=source_counts.values

    #lets plot a pie chart
    plt.pie(counts,labels=labels,autopct='%1.1f%%')

    #add a title to the plot
    plt.title('PIE CHART FOR TWEET SOURCE')
    
    plt.show()
def visualize_sentiments(df):
    sentiment_counts=df['sentiment'].value_counts()
    f,ax=plt.subplots(figsize=(5,5))
    plt.bar([1,2,3],sentiment_counts.values)
    labels={0:'Neutral',1:'Positive',-1:'Negative'}
  
    plt.xticks([1.5,2.5,3.5],[labels[i] for i in sentiment_counts.index])
    plt.title('Bar Plot of Sentiment Counts')
    
    plt.show()
def download_tweets():  # function to download tweets
    auth=authenticate(consumerkey,consumersecret,accesstoken,accesstokensecret)
    api=tweepy.API(auth)
    
    #getting tweets of user DataScienceGH
    tweets=api.search('@stonebwoyb',rts=1,count=500)

    #searching for tweets containing this word: DataScience
   # tweets=api.search('DataScience',count=100)
    return(tweets)
    


if __name__=='__main__':
    tweets=download_tweets()
    df=tweets_to_df(tweets)
    #Uncomment any of the commands below and run the code. You will see the output
   # print(df.head(5))
    #tweets_stats(df)
   #visualize_likes(df)
    #visualize_all(df,['sentiment'])
    #visualize_tweet_source(df)
    #visualize_sentiments(df)
  
    
