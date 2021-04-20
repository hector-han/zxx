# coding:utf-8

import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.options

from utils.login import MainHandler, LoginHandler
from utils.query import QuerySummary, QueryAllTweets
from utils.query_topic import QuerySummaryNew

settings = {
    'template_path': 'template',
    'static_path': 'static',
    "cookie_secret": "143ddfeff1ee42027c849019dbcf69ec",
    "login_url": "/login",
}


if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/login", LoginHandler),
        (r"/query/summary", QuerySummary),
        (r"/query/summary_new", QuerySummaryNew),
        (r"/query/all_tweets", QueryAllTweets),
    ], **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(6789)
    tornado.ioloop.IOLoop.current().start()