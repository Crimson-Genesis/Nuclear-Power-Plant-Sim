import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lib.lib import *

import time, json, threading, random
import zmq
import yaml

class PrimaryCoolent:
    def __init__(self,):
        with open(config_file_name(), "r") as file:
            self.config = yaml.safe_load(file)

        # pprint(self.config)
        self.ctx = zmq.Context()
        self.ctrl_endpoint = self.config["connections"]["ctrl"]["endpoint"]
        self.tick_endpoint = self.config["connections"]["tick"]["endpoint"]
        self.heartbeat_endpoint = self.config["connections"]["heartbeat"]["endpoint"]
        self.telemetry_endpoint = self.config["connections"]["telemetry"]["endpoint"]

        self.ctrl = self.ctx.socket(
            getattr(zmq, self.config["connections"]["ctrl"]["type"])
        )
        self.name = get_name()
        self.ctrl.setsockopt_string(zmq.IDENTITY, self.name)
        # self.ctrl.connect(self.ctrl_endpoint)
        get_connection_object(self.ctrl, self.config["connections"]["ctrl"]["type"])(self.ctrl_endpoint)

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
        self.running = False

    def establish_connections(self):
        self.running = True
        get_connection_object(self.tick, self.config["connections"]["tick"]["type"])(self.tick_endpoint)
        get_connection_object(self.telemetry, self.config["connections"]["telemetry"]["type"])(self.telemetry_endpoint)
        get_connection_object(self.heartbeat, self.config["connections"]["heartbeat"]["type"])(self.heartbeat_endpoint)
        self.tick_thread.start()
        return "Tick, Telemetry and HeartBeat"

    def start(self):
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.dealerListen_thread = threading.Thread(target=self._dealerListener, daemon=True)
        self.dealerListen_thread.start()
        self.tick_thread = threading.Thread(target=self._tick_loop, daemon=True)

        while True:
            time.sleep(1)

    def _tick_loop(self):
        while self.running:
            try:
                data = self.tick.recv_json(flags=zmq.NOBLOCK)
                # self.send_msg(data, type_="tick-check")
                # real_sleep = self.time_step / max(self.time_scale, 1e-9)
                time.sleep(1)
            except zmq.Again:
                time.sleep(0.05)  # poll for control commands while paused

    def _control_loop(self):
        self.send_msg("Okey", type_="check", data=random.randint(1, 1000))
        self.running = True
        while self.running:
            time.sleep(2)

    def _heartbeat_loop(self):
        ...

    def _dealerListener(self):
        while True:
            try:
                identity, msg = self.ctrl.recv_multipart(flags=zmq.NOBLOCK)
                if identity or msg:
                    decoded_msg = json.loads(msg.decode())
                    if (
                        decoded_msg["type"] == "command"
                        and decoded_msg["name"] == "control_system"
                    ):
                        self._executation(decoded_msg)
            except zmq.Again:
                pass
            time.sleep(0.05)

    def _executation(self, msg):
        cmd = msg["command"]
        if cmd == "establish_connections":
            cc = self.establish_connections()
            self.send_msg(
                f"Establish-ED Connections of {cc} at {self.name}", type_="status"
            )

        else:
            self.send_msg("Error", ok=False, error="unknown command")

    def send_msg(self, msg, type_="status", **kwargs):
        data = {
            "type": type_,
            "name": self.name,
            "status": f"{'Running' if self.running else 'Not Running'}",
            "msg": msg,
            **kwargs,
        }
        self.ctrl.send_multipart([b"", json.dumps(data).encode()])


if __name__ == "__main__":
    try:
        pc1 = PrimaryCoolent()
        pc1.start()
    except KeyboardInterrupt:
        print("Keyboard Interrupt !!!")
    except EOFError:
        print("EOF Error !!!")

