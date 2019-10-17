from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

analyser = SentimentIntensityAnalyzer()
conf = config.Conf("config.json").getConf()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

def cal_sentiment(tweet):
    if conf['sentimentMethod'] == "baseonscore":
        if tweet['score']['text']['neu'] > conf['neutralThreshold']:
            return 0
        return tweet['score']['text']['pos'] - tweet['score']['text']['neg']

    if conf['sentimentMethod'] == "compound":
        return tweet['score']['text']['compound']

def sum_of_sentiment(dict_data):
    sum = 0
    for items in dict_data:
        sum += items[1]
    return sum


def plot_sentiment(all_data):
    negs = {}
    poss = {}
    neus = {}
    for key in all_data:
        for tweet in all_data[key]:
            date = tweet['datetime'][0:10]
            if date not in negs:
                negs[date] = 0
            if date not in poss:
                poss[date] = 0
            if date not in neus:
                neus[date] = 0
            st = cal_sentiment(tweet)
            if st == 0:
                neus[date] += 1
            elif st > 0:
                poss[date] += 1
            else:
                negs[date] += 1
    fig1 = plt.figure('time graph')
    neus = sorted(neus.items(), key = lambda x:x[0])
    poss = sorted(poss.items(), key = lambda x:x[0])
    negs = sorted(negs.items(), key = lambda x:x[0])
    dates = [items[0] for items in negs]
    neu_c = [items[1] for items in neus]
    neg_c = [items[1] for items in negs]
    pos_c = [items[1] for items in poss]

    x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    plt.plot(x, pos_c, 'r-o', label='positive')
    plt.plot(x, neg_c, 'b-.o', label='negative')
    plt.plot(x, neu_c, 'k--o', label='neutral')
    fig1.legend()
    plt.gcf().autofmt_xdate()
    #plt.xticks(rotation='vertical', fontsize = 8)

    fig1 = plt.figure('pie graph')
    labels = ['positive','negative', 'neutra']
    sizes = [sum_of_sentiment(poss), sum_of_sentiment(negs), sum_of_sentiment(neus)]
    explode = (0,0,0)
    plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    #startangle表示饼图的起始角度
    plt.show()

def plot_tweet_num(all_data):
    data_via_date = {}
    for key in all_data:
        for tweet in all_data[key]:
            date = tweet['datetime'][0:10]
            if date not in data_via_date:
                data_via_date[date] = 1
            else:
                data_via_date[date] += 1

    data_via_date = sorted(data_via_date.items(), key = lambda x:x[0])

    dates = [items[0] for items in data_via_date]
    count = [items[1] for items in data_via_date]

    x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    plt.plot(x, count, 'r-d')
    plt.gcf().autofmt_xdate()







