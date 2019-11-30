import json


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False


if __name__ == '__main__':
    file = r'D:\code\github\zxx\data\all_tweets_china.jl'
    file2 = r'D:\code\github\zxx\data\all_tweets_china_2.jl'
    whole ={}
    with open(file, encoding='utf-8') as fin, open(file2, 'w', encoding='utf-8') as fout:
        for line in fin.readlines():
            if is_chinese(line):
                print('汉字{}'.format(line))
                continue
            data = json.loads(line)
            ID = data['ID']
            if ID in whole.keys():
                print(data['datetime'], '重复')
                continue
            if data['text'].lower().find('china') >= 0:
                whole[ID] = 1
                fout.write(line)

