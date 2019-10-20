import tornado.web
import json
import os
import time
from wordcloud import WordCloud
from utils.login import BaseHandler
from utils.dbop import query_summary, query_hash_tags, query_tweets_cnt, query_tweets_list


file_path = os.path.dirname(__file__)



class QuerySummary(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        start_time = self.get_argument('start_time')
        end_time = self.get_argument('end_time')
        print(start_time, end_time)
        dates, values = query_summary(start_time, end_time)
        if len(dates) == 0:
            response = {'status': -1, 'msg': '没有数据'}
        else:
            data = {'dates': dates, 'values': values}
            response ={'status': 0, 'msg': 'success', 'data': data}
            hash_tags_frequency = query_hash_tags(start_time, end_time)
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

        total = query_tweets_cnt(start_time, end_time)
        if total == 0:
            response = {'total': 0}
        else:
            rows = query_tweets_list(start_time, end_time, limit, offset, sorted_by)
            response = {'total': total, 'rows': rows}
        self.write(json.dumps(response))