import tornado.web
import json
import os
import time
from wordcloud import WordCloud
from utils.login import BaseHandler
from utils.dbop import query_summary, query_hash_tags, query_tweets_cnt, query_tweets_list


file_path = os.path.dirname(__file__)

map1 = {
    'POSITIVE': '积极',
    'CENTRAL': '中性',
    'NEGATIVE': '消极',
}

color_config = {
    'POSITIVE': '#FF0000', # 红色
    'CENTRAL': '#8B008B', #黑色
    'NEGATIVE': '#0000FF', #蓝色
}

def build_series(values, sentiment):
    series_data = {
        'name': map1[sentiment],
        'type': 'line',
        'itemStyle': {
            'normal': {
                'color': color_config[sentiment],
                'lineStyle': {
                    'color': color_config[sentiment]
                }
            }
        },
        'symbol': None,
        'data': values
    }

    return series_data


def build_resp(data, sentiment):
    dates = list(sorted(data.keys()))
    legends = []
    series = []
    if sentiment == "-1":
        for senti in ['POSITIVE', 'CENTRAL', 'NEGATIVE']:
            legends.append(map1[senti])
            values = []
            for date in dates:
                values.append(data[date][senti])
            series.append(build_series(values, senti))
    else:
        legends.append(map1[sentiment])
        values = []
        for date in dates:
            values.append(data[date][sentiment])
        series.append(build_series(values, sentiment))
    return dates, legends, series


class QuerySummary(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        start_time = self.get_argument('start_time')
        end_time = self.get_argument('end_time')
        cate = self.get_argument('cate')
        sentiment = self.get_argument('sentiment')
        print(start_time, end_time)
        db_data = query_summary(start_time, end_time, cate, sentiment)
        if len(db_data.keys()) == 0:
            response = {'status': -1, 'msg': '没有数据'}
        else:
            dates, legends, series = build_resp(db_data, sentiment)
            data = {'dates': dates, 'legends': legends, 'series': series}
            response = {'status': 0, 'msg': 'success', 'data': data}
            hash_tags_frequency = query_hash_tags(start_time, end_time, cate)
            if len(hash_tags_frequency) > 0:
                wc = WordCloud(background_color='white', width=360, height=360)
                wc.generate_from_frequencies(hash_tags_frequency)

                for f in os.listdir(file_path+'/../static/images/temp'):
                    full_path = os.path.join(file_path+'/../static/images/temp', f)
                    if os.path.isfile(full_path):
                        os.remove(full_path)

                now = time.time()
                fn = str(now) + '.png'
                wc_png_path = os.path.join(file_path+'/../static/images/temp', fn)
                wc.to_file(wc_png_path)
                response['img_src'] = 'static/images/temp/' + fn
            else:
                response['img_src'] = ''

        self.write(json.dumps(response))


tweets_fields = ['id', 'date_time', 'user_id', 'text', 'hash_tags', 'url', 'nbr_retweet', 'nbr_favorite']
class QueryAllTweets(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        print(self.request.body.decode())
        start_time = self.get_argument('start_time')
        end_time = self.get_argument('end_time')
        limit = int(self.get_argument('limit'))
        offset = int(self.get_argument('offset'))
        sorted_by = self.get_argument('sorted_by')
        cate = self.get_argument('cate')
        sentiment = self.get_argument('sentiment')

        total = query_tweets_cnt(start_time, end_time, cate, sentiment)
        if total == 0:
            response = {'total': 0}
        else:
            rows = query_tweets_list(start_time, end_time, limit, offset, sorted_by, cate, sentiment)
            response = {'total': total, 'rows': rows}
        self.write(json.dumps(response))