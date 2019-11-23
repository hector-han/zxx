import pandas as pd
import json
import random

from scripts.settings import stop_words_2
from utils.nlp_models.tools.constants import ch2en
from utils.nlp_models.tools.dictionary import CommonDictionary
from utils.nlp_models.tools.model_snippets import reade_fn_as_label


def export_to_excel_file(data_frame: pd.DataFrame, file_name: str):
    excel_writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    data_frame.to_excel(excel_writer)
    excel_writer.save()


def jl_2_excel():
    excel_file = '../../data/tmp.xlsx'
    jl_file = '../../data/情感标注-汇总.jl'
    df_data = pd.read_json(jl_file, orient='records', encoding='utf-8', lines=True)
    df_data['content'] = df_data['content'].apply(lambda x: ' '.join(x))
    df_data['tid'] = df_data['tid'].astype(str)
    export_to_excel_file(df_data, excel_file)


def excel_to_jl():
    """
    处理标注数据，把excel的数据转换为json line格式的数据
    :return:
    """
    excel_file = '../../data/情感标注-汇总.xlsx'
    jl_file = '../../data/情感标注-汇总.jl'
    distribute = {}
    with open(jl_file, 'w', encoding='utf-8') as fout:
        for cate in ['0', '1', '2', '3', '4']:
            df_data = pd.read_excel(excel_file, sheet_name=cate)
            for i, row in df_data.iterrows():
                if pd.isna(row['情感标注']):
                    continue
                label = ch2en[row['情感标注'].strip()]
                tid = str(row['tweetid'])
                ori_content = row['清洗后推文'].replace('/', ' ').replace('$', '$ ').replace('-', ' ').split(' ')
                content = []

                if label not in distribute:
                    distribute[label] = {}
                word_cnt = distribute[label]

                for w in ori_content:
                    w = w.replace(',', '').replace('.', '').replace('"', '').replace('\'', '').replace(':', '')\
                        .replace('?', '').replace('!', '')
                    if w == '':
                        continue
                    if w in stop_words_2:
                        continue

                    content.append(w)
                    if w not in word_cnt:
                        word_cnt[w] = 0
                    word_cnt[w] += 1
                key_words = row['关键词'] if pd.notna(row['关键词']) else ''
                tmp = {'tid': tid, 'content': content, 'label': label, 'cate': cate, 'keywords': key_words}
                fout.write(json.dumps(tmp))
                fout.write('\n')

    for key, val in distribute.items():
        words = sorted(val.items(), key=lambda x: x[1], reverse=True)
        print('{}前10的词：{}'.format(key, '|'.join([str(e) for e in words[0:10]])))


def split_train_valid():
    all_data = '../../data/data_set_rmdeaf.json'
    # senti_file = '../../data/df_combine.jl'
    train_file = '../../data/train.jl'
    valid_file = '../../data/valid.jl'

    with open(all_data, 'r', encoding="utf8") as fp:
        corpus_list = fp.readlines()

    random.shuffle(corpus_list)
    nb_corpus = len(corpus_list)
    train_cnt = nb_corpus * 0.9

    with open(train_file, 'w', encoding="utf8") as fp_train, open(valid_file, 'w', encoding="utf8") as fp_valid:
        for i, line in enumerate(corpus_list):
            rand = random.uniform(0, nb_corpus)
            if rand <= train_cnt:
                fp_train.write(line)
            else:
                fp_valid.write(line)


def build_dictionary():
    dictionary = CommonDictionary(min_count=10, text_indicator=['content'])
    corpus_dir = r'../../data/'
    dict_dir = r'../../data/vocab'
    train_file = corpus_dir + 'train.jl'
    valid_file = corpus_dir + 'valid.jl'
    print('0000')
    dictionary.read_corpus(reade_fn_as_label(train_file, corpus_dir=corpus_dir))
    dictionary.read_corpus(reade_fn_as_label(valid_file, corpus_dir=corpus_dir))
    print('1111')
    dictionary.save_dictionary(dict_dir=dict_dir)
    print('2222')
    # rnn_dictionary.load_update_dictionary(dict_dir=dict_dir)
    print(dictionary.w2index[dictionary.UNK], dictionary.max_seq_len)
    print(dictionary.w2index.__len__())


def read_from_json_file(file_name: str, lines=True) -> pd.DataFrame:
    """
    从json文件中读取数据到data_frame中
    :param file_name:
    :param lines: 是否启用多行
    :return: pd.DataFrame
    """
    return pd.read_json(file_name, orient='records', encoding='utf-8', lines=lines)


def join_data():
    lda_file = r'../../data/all_tweets_lda_5.jl'
    # cleaned_file = r'../../data/all_tweets_cleaned.jl'
    label_file = r'../../data/df_combine.jl'
    all_data = '../../data/情感标注-汇总.jl'

    df_lda = read_from_json_file(lda_file)
    # df_clean = read_from_json_file(cleaned_file)
    df_label = read_from_json_file(label_file)

    df_lda = df_lda[['ID', 'lda5', 'cleaned']]
    df_label = df_label[df_label.choose == 1]
    df_label = df_label[['ID', 'senti']]

    df_tmp = df_lda.join(df_label.set_index('ID'), on='ID', how='inner')
    df_tmp['content'] = df_tmp.cleaned.apply(lambda x: x.split(' '))
    df_tmp['label'] = df_tmp.senti
    df_tmp.to_json(all_data, orient='records', lines=True)



if __name__ == '__main__':
    # 第一步，先把excel数据转换为json line数据
    # excel_to_jl()
    # jl_2_excel()
    # 第二步， 拆分成训练集合验证集
    # join_data()
    # split_train_valid()
    # 第三步， 构造词典
    build_dictionary()