import util
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
import plotbaseontime as plttm


if __name__ == "__main__":
    infilename = 'data/all_tweets.txt'
    afterfilename = 'data/all_tweets_clean.txt'
    plot_data_file = util.conf['plot_data_file']
    word_frequency_file = 'output/wc.csv'
    mat_con_filename = 'output/con_matrix.csv'

    flag = util.conf['stage']
    if flag == 1:
        util.trans_input_clean(infilename, afterfilename)
    elif flag == 2:
        text_indicator = util.conf['use']

        all_date = util.load_data(afterfilename)
        word_count = {}
        num_all_words = 0

        lst_dataSet = []
        lst_words = []
        for user_id in all_date:
            for data in all_date[user_id]:
                lst_dataSet.append(data[text_indicator].split(' '))
                for word in data[text_indicator].split(' '):
                    util.increase_one(word_count, word)
                    num_all_words += 1
        util.write_to_file(word_count, word_frequency_file)

        #calculate frequency and summary static
        selected_words =[]
        num_of_selected_words = 0
        for items in sorted(word_count.items(), key = lambda x:x[1], reverse=True):
            if items[1] >= util.conf['frequencyThreshold']:
                selected_words.append(items)
                num_of_selected_words += items[1]
                lst_words.append(items[0])
            if len(selected_words) > util.conf['maxWord']:
                break
        print("====slected words are:\n",selected_words)
        print("====select {} words out of {} total words".format(num_of_selected_words, num_all_words))

        mat_con = util.con_frequency(lst_dataSet, lst_words)
        util.write_matri2file(mat_con, lst_words, mat_con_filename)
    elif flag == 3:
        lst_words, mat_con = util.load_con_matrix(mat_con_filename)
        X = util.get_condensed_distance_mat(-np.log(mat_con))
        #X = util.get_condensed_distance_mat(1 - mat_con)
        Z = linkage(X, 'ward')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)

        w_cluster = util.get_clusters_from_linkage(Z, lst_words, util.conf['num_of_cluster'])
        for c in w_cluster:
            print(c)
        plt.show()
        all_date = util.load_data(afterfilename)
        util.split_tweets_into_clusters(all_date, w_cluster, '.\output\cluster')
    elif flag == 4:
        all_date = util.load_data(plot_data_file)
        plttm.plot_sentiment(all_date)
    elif flag == 5:
        #util.combine_folder_remove_dup(r'C:\Users\hengk\PycharmProjects\zxx\data')
        all_data = util.load_data(infilename)
        plttm.plot_tweet_num(all_data)
        plt.show()





