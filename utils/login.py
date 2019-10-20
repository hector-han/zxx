# coding:utf-8

import tornado.web
import tornado.httpserver
from utils.dbop import check_user


class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("user", max_age_days=7)


class MainHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render('main.html')


class LoginHandler(BaseHandler):
    def get(self):
        self.render('index.html', info="")

    def post(self):
        user_name = self.get_body_argument("user_name", default='')  # 获取账号
        passwd = self.get_body_argument("password", default='')  # 获取密码
        if not check_user(user_name, passwd):
            self.render('index.html', info='登录失败')
        else:
            self.set_secure_cookie("user", user_name, expires_days=10)
            self.render("main.html")



