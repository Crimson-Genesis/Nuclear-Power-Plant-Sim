#!/usr/bin/env python

import zmq
import time
import random


def main():
    ed = "ipc:///tmp/main.ctrl"
    cox = zmq.Context()
    soc = cox.socket(zmq.SUB)
    soc.connect(ed)
    soc.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            data = soc.recv_json()
            print(data)
        except zmq.Again:
            ...
        time.sleep(1)

main()
