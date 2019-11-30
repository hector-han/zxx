import json
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


def get_vader_score():
    tweet_file = r'D:\code\github\zxx\data\all_tweets_cleaned.jl'
    tweet_senti_file = r'D:\code\github\zxx\data\all_tweets_vader_senti.jl'
    import pandas as pd
    scores= []
    with open(tweet_file, encoding='utf-8') as fin, open(tweet_senti_file, 'w', encoding='utf-8') as fout:
        for line in fin.readlines():
            data = json.loads(line)
            cleaned = data['cleaned']
            senti_socre = sia.polarity_scores(cleaned)
            data['senti_score'] = senti_socre
            scores.append(senti_socre['compound'])
            fout.write(json.dumps(data))
            fout.write('\n')
    df = pd.DataFrame(scores, columns=['compound'])
    df['compound'].hist().get_figure().savefig(r'D:\code\github\zxx\data\compound_hist.png')


def infer():
    senti_file = r'D:\code\github\zxx\data\all_tweets_vader_senti.jl'
    df_senti = pd.read_json(senti_file, orient='records', lines=True)
    df_senti['datetime'] = df_senti['datetime'].astype(str)

    def _get_senti(scores):
        if scores['compound'] > 0.75:
            return 'POSITIVE'
        if scores['compound'] < -0.75:
            return 'NEGATIVE'
        return 'NEUTRAL'

    df_senti['senti'] = df_senti['senti_score'].apply(_get_senti)

    for senti in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        print(senti, sum((df_senti['senti'] == senti).astype(int)))

    df_senti.to_json(senti_file, orient='records', lines=True)


def merge_lda5():
    senti_file = r'D:\code\github\zxx\data\all_tweets_vader_senti.jl'
    df_senti = pd.read_json(senti_file, orient='records', lines=True)
    lda5_file = r'D:\code\github\zxx\data\all_tweets_lda_5.jl'
    df_lda5 = pd.read_json(lda5_file, orient='records', lines=True)
    df_lda5 = df_lda5[['ID', 'lda5']]
    print(df_senti.shape)
    df_senti = df_senti.join(df_lda5.set_index('ID'), on='ID', how='left')
    df_senti['datetime'] = df_senti['datetime'].astype(str)
    print(df_senti.shape)
    df_senti.to_json(senti_file, orient='records', lines=True)


if __name__ == '__main__':
    # get_vader_score()
    # infer()
    merge_lda5()