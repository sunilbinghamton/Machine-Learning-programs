__author__ = 'Sunil'
'''
  Usage     : Name of stocks to get sentiment index (buy/sell/hold)
  Argument  : Give company stock symbols as a list of string
  Example   : python stock_prediction.py 'AAPL','YHOO','NVDA'
'''

import feedparser
import pandas as pd
import datetime
from pandas_datareader.data import DataReader
import unirest
import sys

# Macros definition
DEBUG = 0
LOCAL_SEARCH = False

# Get company stock symbol from company name
# Use the .csv file to search for company name and its corresponding ticker
NSE_CSV_FILE = 'nseequity.csv'
NASDAQ_CSV_FILE = 'nasdaqcompanylist.csv'
nasdaq_file = pd.read_csv(NASDAQ_CSV_FILE)
symbols = list(nasdaq_file.Symbol)
company_names = list(nasdaq_file.Name)
if DEBUG : print "All listed Nasdaq symbols : \n ", symbols

# Function definitions
def get_feeds(symbol):
  '''
   Perform online feed search for the requested stock symbol.
   Symbol should be valid. Else no feed reponse is received
  '''
  Feeds = []
  Time_Stamps =[]
  stock_feed = feedparser.parse(symbol)
  for entries in stock_feed.get('entries'):
    Feeds.append(entries['title'].encode('ascii','ignore'))
    Time_Stamps.append(entries['updated_parsed'])
  return Feeds, Time_Stamps


def search_for_companies(feed):
  '''
   Perform search of company symbol from company name
  '''
  comp_sym=[]
  for word in feed:
    if word in company_names:
      comp_sym.append(list(word,symbols[company_names.index(word)]))

  return comp_sym

def get_sentiment_meter_from_text_processing_site(feed):
  '''
  Get sentiment meter analysis index values from open source text processing sites
  '''
  import urllib
  data = urllib.urlencode({"text": feed})
  u = urllib.urlopen("http://text-processing.com/api/sentiment/", data)
  sentimeter = u.read()
  return sentimeter


def get_sentiment_meter(feed):
  '''
  Perform sentiment meter analysis from a Sentiment Word Dictionary
  '''
  import re
  pos_sent = 0.0
  neg_sent = 0.0
  neu_sent = 0.0
  for word in feed.split():
    word = re.sub(r'\W+', '',word)
    if len(word) > 1:
      stock_sent = open("SentiWordNet_3.txt",'rb')

      for sent_line in stock_sent:
        sent_line = sent_line.split()
        for sent_word in sent_line:
         if '#' in sent_word and word.lower()+'#' in sent_word:
          pos_index = sent_line[2]
          neg_index = sent_line[3]
          pos_sent += float(pos_index)
          neg_sent += float(neg_index)
          if pos_sent + neg_sent == 0.0:
            neu_sent+=1

      stock_sent.close()
  total = pos_sent + neg_sent + neu_sent
  return pos_sent/total, neg_sent/total, neu_sent/total


def get_previous_data():
  all_data = {}
  date_start = datetime.datetime(2015, 1, 1)
  date_end = datetime.datetime(2015, 5, 1)
  for ticker in ['AAPL', 'IBM', 'YHOO', 'MSFT']:
      all_data[ticker] = DataReader(ticker, 'yahoo', date_start, date_end)
  price = pd.DataFrame({tic: data['Adj Close']
                        for tic, data in all_data.items()})

def main(argvs):
  '''
  This is the main function of the program.
  Arguments : Name of stocks to get sentiment index (buy/sell/hold)
  Usage: Give company stock symbols as a list of string
    Example : 'AAPL','YHOO','NVDA'
  return:  Exit status code
  '''

  # Parse RSS feeds for news related to stocks
  MY_STOCKS = ['AAPL','YHOO','NVDA']
  m = 'http://www.moneycontrol.com/rss/latestnews.xml'
  bs = 'http://www.moneycontrol.com/rss/buzzingstocks.xml'
  bus = 'http://www.moneycontrol.com/rss/business.xml'
  nasdaq_feed = 'http://articlefeeds.nasdaq.com/nasdaq/symbols?symbol='

  # Get all Feed updates
  for symbol in argvs:
    Feeds, Timestamps = get_feeds(bs+symbol)
    print "Feeds for symbol :", symbol , ":\n", Feeds

    # Try to interpret sentence and prediction of stock fluctuation
    # Perform some search operation to predict
    ss_len = float(len(Feeds))
    pos_sentcount = 0
    neg_sentcount = 0
    neu_sentcount = 0

    if ss_len != 0:
      for feed in Feeds:
        try:
          response = unirest.get("https://alchemy.p.mashape.com/text/TextGetTextSentiment?outputMode=json&showSourceText=false&text="+feed,
            headers={ "X-Mashape-Key": "fdrDdillR2mshm8yJZ2MrtsWLZF0p1JFOeAjsnFpyF9RzgKB9b",
                      "Accept": "text/plain" } )

          sentiment = response.body.get('docSentiment')
          if DEBUG : print 'docSentiment : ', response.body.get('docSentiment')

          if sentiment["type"] == "positive" :
            pos_sentcount += 1
          elif sentiment["type"] == "negative" :
            neg_sentcount += 1
          elif sentiment["type"] == "neutral" :
            neu_sentcount += 1
        except TypeError:
          ss_len -= 1
          pass

      if DEBUG : print ss_len, '  ',pos_sentcount, '  ', neg_sentcount
      if ss_len != 0 :
        #calculate probability of overall positive or negative sentiment
        pos = pos_sentcount/ss_len;
        neg = neg_sentcount/ss_len;
        neu = neu_sentcount/ss_len;
        max_sent = max(pos,neg,neu)

        print 'Positive_index : ',pos, 'Negative_index : ',neg, 'Neutral_index : ',neu
        if max_sent == pos :
          print "\nBuy This stock"
        elif max_sent ==  neg :
          print "\nSell this stock"
        else:
          print "\nHold this stock"
        print '\n'

      if ss_len == 0 :
        print "Online sentimeter website not responding..Continuing with local estimate"
        print "Calulating sentiment index.."
        for feed in Feeds:
          pos_sent, neg_sent, neu_sent = get_sentiment_meter(feed)
          #print pos_sent, neg_sent, neu_sent
          pos_sentcount += pos_sent
          neg_sentcount += neg_sent
          neu_sentcount += neu_sent

        #calculate probability of overall positive or negative sentiment
        pos = pos_sentcount/len(Feeds);
        neg = neg_sentcount/len(Feeds);
        neu = neu_sentcount/len(Feeds);
        max_sent = max(pos,neg,neu)

        print 'Positive_index : ',pos, 'Negative_index : ',neg, 'Neutral_index : ',neu
        if max_sent == pos :
          print "\nBuy This stock"
        elif max_sent ==  neg :
          print "\nSell this stock"
        else:
          print "\nHold this stock"
        print '\n'


if __name__ == "__main__":
    main(sys.argv[1:])