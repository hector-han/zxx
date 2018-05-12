import json
import operator
import re

def load_data(filename):
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

def split_tweets(str_tweet):
    return str_tweet.split(' ')

def increase_one(dict_data, key):
    if key in dict_data:
        dict_data[key] += 1
    else:
        dict_data[key] = 1

def write_to_file(dict_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        for items in sorted(dict_data.items(), key = lambda x:x[1], reverse=True):
            line = items[0] + ',' +str(items[1]) + '\n'
            outfile.write(line)

def clean_tweet(tweet):
    # remove \uxxxx
    tmp = tweet.replace('\u2019', "'").lower()
    tmp = re.sub("(http.*\u2026)|(http.*\/[\S]+)|(pic\.twitter.*\/[\S]+)", "", tmp)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z '\t])", " ", tmp).split())


def trans_input_clean(inputfile, outputfile):
    """given a raw data file, clean tweets and save only username, userid, clean tweets, and datatime to outputfile"""
    raw_data = {}
    with open(inputfile, encoding='utf-8') as infile:
        for line in infile:
            tmp = json.loads(line)
            user_id = tmp["user_id"]
            text = tmp["text"]
            text = clean_tweet(text)
            datetime = tmp["datetime"]
            res = {"user_id": user_id, "text":text, "datetime":datetime}

            if user_id in raw_data:
                found = False
                for saved_tweet in raw_data[user_id]:
                    if saved_tweet["text"] == res["text"]:
                        found = True
                        break
                if not found:
                    raw_data[user_id].append(res)
            else:
                raw_data[user_id] = [res]
    with open(outputfile, 'w', encoding='utf-8') as outfile:
        for key in raw_data:
            for tweet in raw_data[key]:
                outfile.write(json.dumps(tweet) + '\n')


if __name__ == "__main__":
    infilename = r'C:\Users\hengk\Desktop\zxx\tweet0322_0415.txt'
    afterfilename = r'C:\Users\hengk\Desktop\zxx\after.txt'
    # trans_input_clean(infilename, afterfilename)
    all_data = load_data_by_date(afterfilename)
    for key in all_data:
        print(key + ',' + str(len(all_data[key])))

    # all_date = load_data(filename)
    # i = 0
    # for items in sorted(all_date.items(), key = lambda x:len(x[1]), reverse=True):
    #     print('======={} send {} tweets=========='.format(items[0], len(items[1])))
    #     for tw in items[1]:
    #         print(tw)
    #     i += 1
    #     if i ==10:
    #         break
    # print(len(all_date))
    # word_count = {}
    # for user_id in all_date:
    #     for data in all_date[user_id]:
    #         for word in split_tweets(data['text']):
    #             real_word = word.replace(' ,.:', '')
    #             real_word = real_word.lower()
    #             if real_word.strip() !=  '':
    #                 increase_one(word_count, real_word)
    # write_to_file(word_count, r'D:\w_c.csv')

