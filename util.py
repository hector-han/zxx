import json
import os
import re
import config
from nltk.corpus import stopwords
import numpy as np
import csv
from scipy.cluster.hierarchy import fcluster

conf = config.Conf("config.json").getConf()
stop_words = stopwords.words('english')

def load_data(filename):
    """
    :param filename:
    :return: a dict with keys = user_id
    """
    all_data = {}
    with open(filename) as infile:
        for line in infile:
            tmp = json.loads(line)
            user_id = int(tmp['user_id'])
            if user_id in all_data:
                all_data[user_id].append(tmp)
            else:
                all_data[user_id] = [tmp]
    return all_data

def load_data_by_date(filename):
    """
    :param filename:
    :return: a dict with keys = date
    """
    all_data = {}
    with open(filename) as infile:
        for line in infile:
            tmp = json.loads(line)
            date = tmp['datetime'][0:10]
            if date in all_data:
                all_data[date].append(tmp)
            else:
                all_data[date] = [tmp]
    return all_data

def increase_one(dict_data, key):
    if key in dict_data:
        dict_data[key] += 1
    else:
        dict_data[key] = 1

def write_to_file(dict_data, filename):
    """
    :param dict_data:
    :param filename:
    :return: null, process write dict to a file, sorted by value of this dict
    """
    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        for items in sorted(dict_data.items(), key = lambda x:x[1], reverse=True):
            line = items[0] + ',' +str(items[1])  + '\n'
            outfile.write(line)

def clean_tweet(tweet):
    # remove \uxxxx
    tmp = tweet.replace('\u2019', "'").lower()
    tmp = re.sub("(http.*\u2026)|(http.*\/[\S]+)|(pic\.twitter.*\/[\S]+)", "", tmp)
    lst_tweet = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z '\t])", " ", tmp).split()
    lst_tweet = [w.lower() for w in lst_tweet]
    lst_tweet_cust = clean_tweet_customise(lst_tweet)
    return ' '.join(lst_tweet), ' '.join(lst_tweet_cust)

def clean_tweet_customise(lst_tweet):
    """
    :param tweet:
    :param config:
    :return: transform the tweet by customised config
    """
    lst_after = []
    for word in lst_tweet:
        word =  word.strip("'")
        if word in stop_words:
            continue
        if word in conf['ignore']:
            continue
        if word in conf['transform']:
            lst_after.append(conf['transform'][word])
        else:
            lst_after.append(word)
    return lst_after

def write_json2file(raw_data, outputfile, isdict = True):
    with open(outputfile, 'w', encoding='utf-8') as outfile:
        if isdict:
            for key in raw_data:
                for tweet in raw_data[key]:
                    outfile.write(json.dumps(tweet) + '\n')
        else:
            for tweet in raw_data:
                outfile.write(json.dumps(tweet) + '\n')

def trans_input_clean(inputfile, outputfile):
    """given a raw data file, clean tweets and save only username, userid, clean tweets, and datatime to outputfile"""
    raw_data = {}
    with open(inputfile, encoding='utf-8') as infile:
        for line in infile:
            tmp = json.loads(line)
            user_id = tmp["user_id"]
            text = tmp["text"]
            clean_text, clean_text_cust = clean_tweet(text)

            datetime = tmp["datetime"]
            res = {"user_id": user_id, "clean_text":clean_text, "clean_text_cust":clean_text_cust, "datetime":datetime}

            if user_id in raw_data:
                found = False
                for saved_tweet in raw_data[user_id]:
                    if saved_tweet["clean_text"] == res["clean_text"]:
                        found = True
                        break
                if not found:
                    raw_data[user_id].append(res)
            else:
                raw_data[user_id] = [res]
    write_json2file(raw_data, outputfile)


def combine_folder2file(loc_folder, out_filename = None):
    """
    :param loc_folder:
    :param out_filename:
    :return:
    """
    basefolder = os.path.dirname(loc_folder)
    print(basefolder)
    if not out_filename:
        out_filename = "res.txt"
    out_filename = os.path.join(basefolder, out_filename)
    with open(out_filename, 'w') as out_file:
        for file in os.listdir(loc_folder):
            with open(os.path.join(loc_folder, file)) as in_file:
                for line in in_file:
                    out_file.write(line + '\n')

def con_frequency(lst_dataSet, lst_words):
    num_of_words = len(lst_words)
    arry_of_freq_words = np.array([0 for i in range(num_of_words)], dtype='int64')
    arry_of_freq_conWords = np.array([[0 for i in range(num_of_words)] for j in range(num_of_words)], dtype='int64')

    for lst_sentence in lst_dataSet:
        for i in range(num_of_words):
            if lst_words[i] in lst_sentence:
                #find word[i]
                arry_of_freq_words[i] += 1
                for j in range(num_of_words):
                    if lst_words[j] in lst_sentence:
                        arry_of_freq_conWords[i,j] += 1
    arry_of_freq_sqrtConv = np.array([[  np.sqrt(a * b)  for a in arry_of_freq_words] for b in arry_of_freq_words], dtype='int64')
    return arry_of_freq_conWords / arry_of_freq_sqrtConv

def write_matri2file(mat_data, header, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as outfile:
        csv_writer = csv.writer(outfile, delimiter=",")
        csv_writer.writerow([" "] + header)
        for i in range(len(mat_data)):
            csv_writer.writerow([header[i]] + list(mat_data[i]))

def get_condensed_distance_mat(dist_mat):
    num = len(dist_mat)
    tmp = []
    for i in range(num):
        for j in range(i+1, num):
            tmp.append(dist_mat[i,j])
    return np.array(tmp)

def load_con_matrix(filename):
    """
    load a con matrix from filename to memory
    :param filename:
    :return: lst_words, con_matrix
    """
    lst_words = []
    con_matrix = []
    with open(filename, encoding='utf-8',newline='') as infile:
        reader = csv.reader(infile, delimiter=',')
        line_no = 0
        for row in reader:
            if line_no == 0:
                lst_words = row[1:]
            else:
                con_matrix.append(row[1:])
            line_no += 1

    con_matrix = np.array(con_matrix, dtype='float')
    return lst_words, con_matrix

def get_clusters_from_linkage(link_matrix, lst_words, num_clusters = 2):
    res = fcluster(link_matrix, num_clusters, criterion='maxclust')
    words_cluster = []
    for i in range(num_clusters):
        words_cluster.append([lst_words[j] for j in range(len(lst_words)) if res[j] == i+1])
    return words_cluster

def which_cluster_belong(tweet, w_cluster):
    num_of_cluster = len(w_cluster)
    indicator = [0 for i in range(num_of_cluster)]
    for word in tweet.split():
        for i in range(num_of_cluster):
            if word in w_cluster[i]:
                indicator[i] += 1
                break
    return indicator.index(max(indicator))

def split_tweets_into_clusters(all_data, w_cluster, folder):
    """
    :param all_data: dict with key=userid, value=list of tweets of this user
    :param w_cluster: list of list of different clusters
    :return:
    """
    use = conf['use']
    tmp = {}
    num_of_cluster = len(w_cluster)
    for i in range(num_of_cluster):
        tmp[i] = []
    for key in all_data:
        for tweet in all_data[key]:
            c_id = which_cluster_belong(tweet[use], w_cluster)
            tmp[c_id].append(tweet)

    for i in range(num_of_cluster):
        filename = os.path.join(folder, 'cluster_{}.txt'.format(i))
        write_json2file(tmp[i], filename, isdict=False)

