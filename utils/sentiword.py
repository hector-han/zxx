import nltk
import numpy as np
import json
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word,tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    if lemma in ['sink', 'recession', 'lose']:
        return [0, 1, 0]
    if lemma in ['win']:
        return [1, 0, 0]

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]


def get_stc_sentiment(pos_val):
    ret = []
    words = [w for (w, t) in pos_val]
    n = len(words)
    for i, (word, tag) in enumerate(pos_val):
        word_senti = get_sentiment(word, tag)
        if len(word_senti) != 3:
            continue
        # print(word_senti)
        s = max(i-2, 0)
        t = min(i+3, n)

        if 'china' in words[s:t]:
            word_senti = [word_senti[1], word_senti[0], word_senti[2]]
        ret.append(word_senti)
    if len(ret) == 0:
        return [0.0, 0.0, 1.0]
    ret = np.sum(ret, axis=0)
    return ret.tolist()


# words_data = ['this','movie','is','wonderful']
# pos_val = nltk.pos_tag(words_data)
# print(get_stc_sentiment(pos_val))


def cal_senti(scores):

    if scores[0] + scores[1] == 0:
        return 'NEUTRAL'

    if scores[2] / (scores[0] + scores[1]) > 15:
        return 'NEUTRAL'

    if scores[1] > scores[0]:
        return 'NEGATIVE'
    if scores[0] > 0.75 and scores[0] > 2 * scores[1]:
        return 'POSITIVE'
    return 'CENTRAL'


if __name__ == '__main__':
    tweet_file = r'D:\code\github\zxx\data\all_tweets_cleaned.jl'
    tweet_senti_file = r'D:\code\github\zxx\data\all_tweets_cleaned_senti.jl'
    import pandas as pd
    scores= []
    with open(tweet_file, encoding='utf-8') as fin, open(tweet_senti_file, 'w', encoding='utf-8') as fout:
        for line in fin.readlines():
            data = json.loads(line)
            cleaned = data['cleaned']
            cleaned = cleaned.split(' ')
            pos_val = nltk.pos_tag(cleaned)
            stc_senti = get_stc_sentiment(pos_val)
            scores.append(stc_senti)
            data['senti_score'] = stc_senti
            fout.write(json.dumps(data))
            fout.write('\n')
    df = pd.DataFrame(scores, columns=['pos', 'neg', 'sub'])
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(df['pos'].describe(percentiles=percentiles))
    print(df['neg'].describe(percentiles=percentiles))
    print(df['sub'].describe(percentiles=percentiles))