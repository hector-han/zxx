import json


if __name__ == '__main__':
    file = r'D:\code\github\data\all_tweets_china.jl'
    file2 = r'D:\code\github\data\all_tweets_china_2.jl'
    whole ={}
    with open(file, encoding='utf-8') as fin, open(file2, 'w', encoding='utf-8') as fout:
        for line in fin.readlines():
            data = json.loads(line)
            ID = data['ID']
            if ID in whole.keys():
                print(data['datetime'], '重复')
                continue
            if data['text'].lower().find('china') >= 0:
                whole[ID] = 1
                fout.write(line)

