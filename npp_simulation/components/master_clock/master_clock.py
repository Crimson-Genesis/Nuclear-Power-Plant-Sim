import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lib.lib import *

import time, json, threading
import zmq
import yaml
import random


class ClockService:
    def __init__(self):

        with open(config_file_name(), "r") as file:
            self.config = yaml.safe_load(file)

        self.name = get_name()
        self.ctx = zmq.Context()
        self.ctrl_endpoint = self.config["connections"]["ctrl"]["endpoint"]
        self.tick_endpoint = self.config["connections"]["tick"]["endpoint"]
        self.heartbeat_endpoint = self.config["connections"]["heartbeat"]["endpoint"]
        self.telemetry_endpoint = self.config["connections"]["telemetry"]["endpoint"]

        self.ctrl = self.ctx.socket(
            getattr(zmq, self.config["connections"]["ctrl"]["type"])
        )
        self.ctrl.setsockopt_string(zmq.IDENTITY, self.name)

        self.tick = self.ctx.socket(
            getattr(zmq, self.config["connections"]["tick"]["type"])
        )

        self.heartbeat = self.ctx.socket(
            getattr(zmq, self.config["connections"]["heartbeat"]["type"])
        )
        self.telemetry = self.ctx.socket(
            getattr(zmq, self.config["connections"]["telemetry"]["type"])
        )

        get_connection_object(self.ctrl, self.config["connections"]["ctrl"]["type"])(
            self.ctrl_endpoint
        )
        get_connection_object(self.tick, self.config["connections"]["tick"]["type"])(
            self.tick_endpoint
        )

        # ------------------------------------------------------------------------------------------------------------------------
        self.sim_time = 0.0
        self.tick_index = 0
        self.time_step = 1.0
        self.time_scale = 1.0
        self.paused = False
        self.running = True
        self.connections = False
        self.heartbeat_interval = 5.0

    def start(self):
        self.contorl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.contorl_thread.start()

        self.dealerListener_thread = threading.Thread(
            target=self._dealerListener, daemon=True
        )
        self.dealerListener_thread.start()

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.tick_therad = threading.Thread(target=self._tick_loop, daemon=True)
        self.tick_therad.start()
        while True:
            time.sleep(1)

    def establish_connections(self):
        get_connection_object(
            self.telemetry, self.config["connections"]["telemetry"]["type"]
        )(self.telemetry_endpoint)
        get_connection_object(
            self.heartbeat, self.config["connections"]["heartbeat"]["type"]
        )(self.heartbeat_endpoint)
        self.heartbeat_thread.start()
        self.connections = True
        return "Telemetry and Heartbeat"

    def _tick_loop(self):
        while not self.connections:
            time.sleep(0.1)
        self.send_msg("Tick Started", type_="status", tick_status="started")
        while self.running:
            if not self.paused and self.time_scale > 0:
                self.tick_index += 1
                self.sim_time += self.time_step
                msg = {
                    "type": "tick",
                    "sim_time": self.sim_time,
                    "tick_index": self.tick_index,
                    "time_step": self.time_step,
                    "time_scale": self.time_scale,
                    "running": True,
                }
                self.tick.send_json(msg)
                # self.send_msg(msg)

                real_sleep = self.time_step / max(self.time_scale, 1e-9)
                time.sleep(real_sleep)
            else:
                time.sleep(0.05)  # poll for control commands while paused

    def _control_loop(self):
        self.send_msg("Okey", type_="check", data=random.randint(1, 1000))
        while True:
            try:
                identity, msg = self.ctrl.recv_multipart(flags=zmq.NOBLOCK)
                jsonify_msg = json.loads(msg.decode())
            except zmq.Again:
                pass
            time.sleep(1)

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

    def _executation(self, jsonify_msg, *args):
        cmd = jsonify_msg["command"]
        if cmd == "pause":
            self.paused = True
            self.send_msg("Paused")
        elif cmd == "resume":
            self.paused = False
            self.send_msg("Resume")
        elif cmd == "scale":
            # val = float(req.get("value", 1.0))
            self.time_scale = jsonify_msg["value"]
            self.send_msg("scalling time", time_scale=self.time_scale)
        elif cmd == "step":
            # cnt = int(req.get("count", 1))
            # # do synchronous steps (while paused)
            # if self.paused:
            #     for _ in range(cnt):
            #         self.tick_index += 1
            #         self.sim_time += self.time_step
            #         msg = {
            #             "type": "tick",
            #             "sim_time": self.sim_time,
            #             "tick_index": self.tick_index,
            #             "time_step": self.time_step,
            #             "time_scale": self.time_scale,
            #             "running": False,
            #         }
            #         self.pub.send_json(msg)
            #     self.rep.send_json({"ok": True, "msg": f"stepped {cnt}"})
            # else:
            #     self.rep.send_json({"ok": False, "error": "not paused"})
            ...
        elif cmd == "status":
            self.send_msg(
                "status info",
                sim_time=self.sim_time,
                time_scale=self.time_scale,
                paused=self.paused,
            )
        elif cmd == "stop":
            self.running = False
            self.send_msg("Stopping")
        elif cmd == "ready":
            self.send_msg("ready or not", ok=True)
        elif cmd == "establish_connections":
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

    def _heartbeat_loop(self): ...


if __name__ == "__main__":
    try:
        clock = ClockService()
        clock.start()
    except KeyboardInterrupt:
        print("Keyboard Interrupt !!!")
    except EOFError:
        print("EOF Error !!!")
