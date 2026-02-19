#!/usr/bin/env python

import os
import sys
import subprocess
import yaml
import threading
import zmq
from pydantic import BaseModel, ValidationError
from typing import List
from pprint import pprint
import time, random, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lib.lib import *


class Config(BaseModel):
    name: str
    version: int
    features: List[str]


class Runner:
    def __init__(self):
        with open(config_file_name(), "r") as file:
            self.config = yaml.safe_load(file)
        # pprint(self.config)

        self.ctx = zmq.Context()

        self.ctrl_endpoint = self.config["connections"]["ctrl"]["endpoint"]
        self.tick_endpoint = self.config["connections"]["tick"]["endpoint"]
        self.heartbeat_endpoint = self.config["connections"]["heartbeat"]["endpoint"]
        self.telemetry_endpoint = self.config["connections"]["telemetry"]["endpoint"]
        self.spawn_list = self.config["spawn_list"]
        self.proc_dict = {"background": dict(), "foreground": dict()}

        self.ctrl = self.ctx.socket(
            getattr(zmq, self.config["connections"]["ctrl"]["type"])
        )
        self.name = get_name()
        self.ctrl.setsockopt_string(zmq.IDENTITY, self.name)

        self.tick = self.ctx.socket(
            getattr(zmq, self.config["connections"]["tick"]["type"])
        )
        self.tick.setsockopt_string(zmq.SUBSCRIBE, "")
        self.heartbeat = self.ctx.socket(
            getattr(zmq, self.config["connections"]["heartbeat"]["type"])
        )
        self.telemetry = self.ctx.socket(
            getattr(zmq, self.config["connections"]["telemetry"]["type"])
        )

        get_connection_object(self.heartbeat, self.config["connections"]["heartbeat"]["type"])(
            self.heartbeat_endpoint
        )

        self.shutdown = False

    def establish_connections(self):
        get_connection_object(self.tick, self.config["connections"]["tick"]["type"])(self.tick_endpoint)
        get_connection_object(self.telemetry, self.config["connections"]["telemetry"]["type"])(self.telemetry_endpoint)
        self.heartbeat_thread.start()
        return "Tick and Telemetry"

    def start(self):
        try:
            self.foreground_spawner()
            get_connection_object(
                self.ctrl, self.config["connections"]["ctrl"]["type"]
            )(self.ctrl_endpoint)
            # self.ctrl.connect(self.ctrl_endpoint)
            self.background_spawner()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            time.sleep(1)
            self.send_msg("Start Establish-ING Connections", type_="command", command="establish_connections")
            while True:
                if self.shutdown:
                    print("Good Bye 😄")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            self.terminate_background_spawns()

    def background_spawner(self):
        for proc in self.spawn_list["background"]:
            self.proc_dict["background"][proc["name"]] = subprocess.Popen(
                ["python", *proc["args"], get_file(proc["path"])],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.send_log_msg(f"{proc['name']} - Started")
            time.sleep(0.5)

    def subProc_dataMoniter(self):
        while True:
            for proc_name, proc in self.proc_dict["background"].items():
                procout = [i for i in proc.stdout]
                procerr = [i for i in proc.stderr]
                if procout and procerr:
                    self.send_msg(f"Name: {proc_name} | Out: {procout} | Err: {procerr}", type_="proc_out/err")
                elif procout:
                    self.send_msg(f"Name: {proc_name} | Out: {procout}", type_="proc_out/err")
                elif procerr:
                    self.send_msg(f"Name: {proc_name} | Err: {procerr}", type_="proc_out/err")
            time.sleep(0.5)

    def terminate_background_spawns(self):
        for proc_name,proc in self.proc_dict["background"].items():
            proc.terminate()
            self.send_log_msg(f"Termination Complete - {proc_name}")
            time.sleep(0.5)

    def foreground_spawner(self):
        for proc in self.spawn_list["foreground"]:
            self.proc_dict["foreground"][proc["name"]] = subprocess.Popen(
                ["python", *proc["args"], get_file(proc["path"])],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        time.sleep(3)

    def terminate_foreground_spawns(self):
        for proc in self.proc_dict["foreground"].values():
            proc.terminate()
            time.sleep(0.5)

    def _tick_loop(self): ...

    def _control_loop(self):
        self.running = True
        while self.running:
            try:
                identity, msg = self.ctrl.recv_multipart(flags=zmq.NOBLOCK)
                if identity or msg:
                    decoded_msg = json.loads(msg.decode())
                    if decoded_msg["type"] in ("command",):
                        if decoded_msg["command"] == "shutdown":
                            self._termination_start()
                    if decoded_msg["type"] in ("command",) and decoded_msg["command"] == "establish_connections":
                        cc = self.establish_connections()
                        self.send_msg(f"Establish-ED Connections of {cc} at {self.name}", type_="status")
            except zmq.Again:
                pass
            time.sleep(2)

    def send_msg(self, msg, type_="status", **kwargs):
        data = {"type": type_, "name": self.name, "msg": msg, **kwargs}
        self.ctrl.send_multipart([b"", json.dumps(data).encode()])

    def send_log_msg(self, msg: str):
        data = {"type": "log", "name": self.name, "msg": msg}
        self.ctrl.send_multipart([b"", json.dumps(data).encode()])

    def _heartbeat_loop(self):
        while True:
            # print("Heartbeat - Okey")
            time.sleep(2)

    def _termination_start(self):
        self.terminate_background_spawns()
        self.send_log_msg("Background Process Termination Complete...")
        self.send_log_msg("Main Control System will be terminated in 5s.")
        for i in range(5, -1, -1):
            self.send_log_msg(f"{i}s")
            time.sleep(1)
        self.send_msg(
            msg="Control System Shutdown Sequence",
            type_="command",
            command="control-system-shutdown",
        )
        self.shutdown = True


if __name__ == "__main__":
    runner = Runner()
    runner.start()
else:
    ...
