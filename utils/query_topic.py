import tornado.web
import json
import os
import time
from utils.login import BaseHandler
from utils.dbop import query_summary_new


file_path = os.path.dirname(__file__)

map1 = {
    '-1': '未知',
    '0': 'topic_0',
    '1': 'topic_1',
    '2': 'topic_2',
    '3': 'topic_3',
    '4': 'topic_4'
}

color_config = {
    '0': '#FF0000', # 红色
    '1': '#8B008B', #黑色
    '2': '#0000FF', #蓝色
    '3': '#008000',  # 绿色
    '4': '#FF00FF',  # meta
    '-1': '#FFFF00',  # 黄色
}

def build_series(values, topic):
    series_data = {
        'name': map1[topic],
        'type': 'line',
        'itemStyle': {
            'normal': {
                'color': color_config[topic],
                'lineStyle': {
                    'color': color_config[topic]
                }
            }
        },
        'symbol': None,
        'data': values
    }

    return series_data


def build_resp(data):
    dates = list(sorted(data.keys()))
    legends = []
    series = []

    for topic in ['0', '1', '2', '3', '4']:
        legends.append(map1[topic])
        values = []
        for date in dates:
            values.append(data[date][topic])
        series.append(build_series(values, topic))

    return dates, legends, series


class QuerySummaryNew(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        start_time = self.get_argument('start_time')
        end_time = self.get_argument('end_time')
        sentiment = self.get_argument('sentiment')
        print('QuerySummaryNew', start_time, end_time)
        db_data = query_summary_new(start_time, end_time, sentiment)
        if len(db_data.keys()) == 0:
            response = {'status': -1, 'msg': '没有数据'}
        else:
            # if sentiment == "-1":
            #     db_data = _get_percentage_value(db_data)
            dates, legends, series = build_resp(db_data, sentiment)

            data = {'dates': dates, 'legends': legends, 'series': series}
            response = {'status': 0, 'msg': 'success', 'data': data}

        self.write(json.dumps(response))

