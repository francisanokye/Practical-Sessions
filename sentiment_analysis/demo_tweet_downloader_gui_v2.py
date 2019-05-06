import Tkinter as tk
from tkMessageBox import showinfo
import time
#This is for python 3
#import tkinter as tk
#from tkinter.messagebox import showinfo

import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import matplotlib

consumerkey='GhGZJQqiE14UBJl1TV1oj5rKU'
consumersecret='PemV7yRLHxpnYzhh9IhuIzrPWRYalCesrXmBBtxvxoiLDCB0Af'

accesstoken='1033643831793729536-nSVdnzCM5UrHo7qAZAMZ4gpbXDq30Z'
accesstokensecret='t5oHuc2RX5vbWRJJxPZ1LJ2v5gA1q4tN4GJ3f22R8wiSB'

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
    plt.title('TWEET ANALYSIS ON TWEET LIKES')
    plt.show()
    
def visualize_retweets(df):
    time_likes=pd.Series(data=df['retweets'].values,index=df['date'])
    time_likes.plot(figsize=(16,4),color='b',label='likes')
    plt.title('TWEET ANALYSIS ON RETWEET COUNT')
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

def tweets_to_df(tweets):     # function to save tweets to a dataframe
   
    data=[[tweet.id,tweet.full_text,tweet.created_at,tweet.source,tweet.favorite_count,
           tweet.retweet_count,len(tweet.full_text)] for tweet in tweets]
    df=pd.DataFrame(data,columns=['id','tweet','date','source','likes','retweets','length'])

    #sentiment analysis . you can uncomment the line add the sentiment
    df['sentiment']=[analyze_sentiment(tweet) for tweet in df['tweet'].values]
    
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
    plt.title('Bar Plot of Sentiment Counts (%s tweets)'%len(df))

    plt.show()

    
def download_tweets(userid):  # function to download tweets
    auth=authenticate(consumerkey,consumersecret,accesstoken,accesstokensecret)
    api=tweepy.API(auth)
    
    #getting tweets of user DataScienceGH
    #tweets=api.search('@stonebwoyb',rts=1,count=500)
    tweets=tweepy.Cursor(api.user_timeline,screen_name=userid,tweet_mode='extended',since='2019-04-24').items(300)
    
    #searching for tweets containing this word: DataScience
   # tweets=api.search('DataScience',count=100)
    return(tweets)
    

    
class TweetDownloader(tk.Tk):
    def __init__(self,parent):
        tk.Tk.__init__(self,parent)
        self.grid()
       # self.geometry('400x400+250+250')
        self.interface()
        auth=authenticate(consumerkey,consumersecret,accesstoken,accesstokensecret)
        global api
        api=tweepy.API(auth)


    def interface(self):
        entryframe=tk.Frame()
        entryframe.pack(side='left')
        
        tk.Label(entryframe,text='User ID').grid(row=0,column=0,sticky='w')
        self.userid=tk.StringVar()
        self.searchterm=tk.StringVar()
        self.analyze_component=tk.StringVar()
        tk.Entry(entryframe,textvariable=self.userid).grid(row=0,column=1)
        
        tk.Label(entryframe,text='Search Term').grid(row=1,column=0,sticky='w')
        tk.Entry(entryframe,textvariable=self.searchterm).grid(row=1,column=1)

        optionsframe=tk.Frame(self)
        optionsframe.pack(side='left')

        tk.Label(optionsframe,text='Choose what to analyze').pack(side='top')
        tk.Radiobutton(optionsframe,text='Likes',variable=self.analyze_component,value='a').pack(side='top',anchor='w')
        tk.Radiobutton(optionsframe,text='Retweets',variable=self.analyze_component,value='b').pack(side='top',anchor='w')
        tk.Radiobutton(optionsframe,text='Sentiment',variable=self.analyze_component,value='c').pack(side='top',anchor='w')
        tk.Radiobutton(optionsframe,text='Source',variable=self.analyze_component,value='d').pack(side='top',anchor='w')


        tk.Button(entryframe,text='OK',command=self.analyze).grid(row=2,column=1,sticky='w',pady=5)
        tk.Button(entryframe,text='QUIT',command=self.destroy).grid(row=2,column=0,sticky='w',pady=5)
    def analyze(self,file_=False):
        userid=self.userid.get()
        searchterm=self.searchterm.get()
        #auth=authenticate(consumerkey,consumersecret,accesstoken,accesstokensecret)
        #global api
       # api=tweepy.API(auth)
        
        if file_==False:
  
            if searchterm=='':
                showinfo('Enter Search Term')
            else:
                tweets=tweepy.Cursor(api.search,q=searchterm,tweet_mode='extended',since='2019-04-25').items(500)
        #tweets=tweepy.Cursor(api.user_timeline,screen_name=userid,tweet_mode='extended',since='2019-04-24').items(300)
            
            self.df=tweets_to_df(tweets)
        else:
            self.df=pd.read_csv('tweetdownload.csv')
       
        self.call_plot()
        
        
        

    def call_plot(self):
        plt.close()
        code=self.analyze_component.get()
        print(code)
        if code=='':
            showinfo('','Select an Option')
        else:
            
            if code=='a':
                visualize_likes(self.df)
            elif code=='b':
                visualize_retweets(self.df)
            elif code=='c':
                visualize_sentiments(self.df)
                
                
            elif code=='d':
                visualize_tweet_source(self.df)
        
           
            #self.after(10,self.initialize_plot)
            
    def initialize_plot(self):
        self.call_plot()
        print('fetching updates')
        

          

if __name__=="__main__":
    app=TweetDownloader(None)
    app.mainloop()
    
