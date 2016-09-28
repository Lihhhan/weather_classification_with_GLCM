#coding=utf-8
import logging

class mylog( ):
    def __init__(self, name):
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename='./log/%s.log'%name,
            filemode='a')
        
    def info(self, s):
        logging.info(s)



