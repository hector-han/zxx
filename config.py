import json

class Conf(object):
    def __init__(self, confFilePath):
        with open(confFilePath, encoding='utf-8') as c_file:
            self.conf = json.load (c_file)

    def getConf(self):
        return self.conf


if __name__ == "__main__":
    conf = Conf(r"C:\Users\hengk\Desktop\zxx\config.json")
    print(conf.getConf())
