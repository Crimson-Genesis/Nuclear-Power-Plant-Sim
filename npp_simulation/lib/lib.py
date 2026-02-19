# funciton to join file to base path

import os
import sys
import time

BASE_PATH = os.path.dirname(os.path.dirname(__file__))

CONNECTION_TYPE = {
    "PUB":"bind",
    "SUB":"connect",
    "DEALER":"connect",
    "ROUTER":"bind",
    "PUSH":"bind",
    "PULL":"connect",
    "REQ":"connect",
    "REP":"bind",
}

def get_file(path):
    return os.path.join(BASE_PATH, path)

def config_file_name():
    return os.path.join(BASE_PATH,"config",f"{sys.argv[0].split('/')[-1].split('.')[0]}.yaml")

def get_name():
    return sys.argv[0].split("/")[-1].split(".")[0]

def get_connection_object(con_obj, type_name):
    return getattr(con_obj, CONNECTION_TYPE[type_name])

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
