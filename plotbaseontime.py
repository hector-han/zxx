from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

analyser = SentimentIntensityAnalyzer()
conf = config.Conf("config.json").getConf()
useForSentiment = conf['useForSentiment']

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

def plot_sentiment(all_data):
    negs = {}
    poss = {}
    for key in all_data:
        for tweet in all_data[key]:
            date = tweet['datetime'][0:10]
            if date not in negs:
                negs[date] = 0
            if date not in poss:
                poss[date] = 0
            snt = analyser.polarity_scores(tweet[useForSentiment])
            if snt['compound'] > 0:
                poss[date] += 1
            else:
                negs[date] += 1

    negs = sorted(negs.items(), key = lambda x:x[0])
    poss = sorted(poss.items(), key = lambda x:x[0])

    dates = [items[0] for items in negs]
    neg_count = [items[1] for items in negs]
    pos_count = [items[1] for items in poss]

    x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    plt.plot(x, neg_count, 'r-', x, pos_count, 'b-.')
    plt.gcf().autofmt_xdate()
    plt.show()







