#!/usr/bin/env python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lib.lib import *

import zmq
import time
import random
import yaml
import json

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.widgets import TextArea
import threading


class ControlSystem:
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
        get_connection_object(self.ctrl, self.config["connections"]["ctrl"]["type"])(
            self.ctrl_endpoint
        )
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

        self.running = True
        # Top Left: read-only log/output
        self.display_pane = TextArea(
            scrollbar=True,
            focusable=False,
            height=20,
            wrap_lines=False,
            # dont_extend_height=True,
            # dont_extend_width=True,
            # read_only=True,
        )

        # Bottom Left: read-only log/output
        self.log_pane = TextArea(
            scrollbar=True,
            focusable=False,
            wrap_lines=False,
            # dont_extend_height=True,
            # dont_extend_width=True,
            # read_only=True,
        )

        # Top Right: Output section
        self.output_pane = TextArea(
            scrollbar=True,
            focusable=False,
            multiline=True,
            # dont_extend_height=True,
            # dont_extend_width=True,
            # read_only=True,
        )

        # Bottom Right: editable input
        self.input_pane = TextArea(
            prompt="> ",
            scrollbar=True,
            focusable=True,
            multiline=False,
            wrap_lines=False,
            # dont_extend_height=True,
            # dont_extend_width=True,
        )

        # Other pane:
        # Top Right: Output section
        # self.proc_output_pane = TextArea(
        #     scrollbar=True,
        #     focusable=False,
        #     multiline=True,
        #     # read_only=True,
        # )

        # Vertical split with separator
        self.root_container = VSplit(
            [
                HSplit(
                    [
                        self.log_pane,
                        Window(height=1, char="─", style="class:line"),
                        self.display_pane,
                    ]
                ),
                Window(width=1, char="│", style="class:line"),  # separator
                HSplit(
                    [
                        # self.proc_output_pane,
                        # Window(height=1, char="─", style="class:line"),
                        self.output_pane,
                        Window(height=1, char="─", style="class:line"),  # separator
                        self.input_pane,
                    ],
                    width=80,
                ),
            ]
        )
        self.layout = Layout(
            container=self.root_container, focused_element=self.input_pane
        )
        self.kb = KeyBindings()
        self._binds()
        self.application = Application(
            layout=self.layout,
            key_bindings=self.kb,
            full_screen=True,
            mouse_support=True,
        )
        self.dealer_registry = dict()
        # self.current_log = dict()
        self.log_registry = dict()
        self.shutdown_process_started = False
        self.established_connections = False

    def start(self):
        self.dealer_listen_thrade = threading.Thread(
            target=self._dealerListener, daemon=True
        )
        self.dealer_listen_thrade.start()
        self.tick_loop_thread = threading.Thread(target=self._tick_loop, daemon=True)

        self.tick_loop_thread.start()
        self._window_start()
        if self.running == False:
            ...

    def establish_connections(self):
        get_connection_object(self.tick, self.config["connections"]["tick"]["type"])(
            self.tick_endpoint
        )
        get_connection_object(
            self.telemetry, self.config["connections"]["telemetry"]["type"]
        )(self.telemetry_endpoint)
        get_connection_object(
            self.heartbeat, self.config["connections"]["heartbeat"]["type"]
        )(self.heartbeat_endpoint)

    def _binds(self):
        @self.kb.add("c-c")
        def exit(event):
            "Quit with Ctrl-C."
            self.running = False
            event.app.exit()

        @self.kb.add("enter")
        def take_input(event):
            text = self.input_pane.text.strip()
            if text:
                # formatted = FormattedText([("class:input_text", f"> {text}\n")])
                self.output_pane.buffer.insert_text(f"> {text}\n")
                self._executation(text)
                self.input_pane.text = ""

    def _executation(self, inStr):
        help = """
    p   - pause     : This will pause the entire simulation system.
    r   - resume    : This will resume the paused system.
    s   - scale     : This will increse the scale of simulation by the factor of 1.
    st  - step      : This will increse the speed of the simulation by the 1x.
    S   - status    : This will show status of all the systems.
    R   - ready     : This will force refresh the display_pane, and force check if all the compontes are ready to start.
    e   - exit      : This will gracefull quit the simulation program.
    ...
        """

        inStr_list = inStr.split(" ")
        if not self.shutdown_process_started:
            if inStr in ("p", "pause"):
                ...
            elif inStr in ("r", "resume"):
                ...
            elif inStr in ("s", "scale"):
                ...
            elif inStr in ("st", "step"):
                ...
            elif inStr in ("S", "status"):
                ...
            elif inStr in ("R", "ready"):
                ...
            elif inStr in ("cc", "clear"):
                # self.display_pane.text = ""
                self.log_pane.text = ""
                self.output_pane.text = ""
            elif inStr == "ref":
                self.application.renderer.clear()
                self.application._redraw()
            elif inStr == "asdf" or inStr == "e" or inStr == "quit" or inStr == "exit":
                self.shutdown_process_started = True
                self.sendCmd("runner", "shutdown", "system shutdown")
                # self.ctrl.send_multipart(
                #     [
                #         "runner".encode(),
                #         b"",
                #         json.dumps(
                #             {
                #                 "type": "command",
                #                 "command": "shutdown",
                #                 "msg": "system shutdown",
                #             }
                #         ).encode(),
                #     ]
                #
            elif inStr in ("h", "help"):
                self.output_pane.buffer.insert_text(help + "\n")
            elif (
                len(inStr_list) >= 2
                and inStr_list[0] == "test"
                and inStr_list[1] in self.dealer_registry.keys()
                and self.established_connections
            ):
                threading.Thread(
                    target=self.test_msg,
                    daemon=True,
                    args=[
                        inStr_list[1],
                        int(inStr_list[2:3][0]) if len(inStr_list[2:3]) else 5,
                        int(inStr_list[3:4][0]) if len(inStr_list[3:4]) else 1,
                    ],
                ).start()
            elif (
                len(inStr_list) >= 2
                and inStr_list[0] == "reactor"
                and inStr_list[1] == "part-sim-start"
                and self.established_connections
            ):
                self.sendCmd(
                    "reactor",
                    "start-partical-sim-thread",
                    "start the partical simulation...",
                )
                print("hello")
            else:
                self.output_pane.buffer.insert_text(
                    f"[ERROR] - type \\help for the argument definition list.\n"
                )
        else:
            if inStr in ("stop"):
                ...

    def sendCmd(self, to_, cmd, msg):
        self.ctrl.send_multipart(
            [
                to_.encode(),
                b"",
                json.dumps(
                    {
                        "type": "command",
                        "command": cmd,
                        "name": self.name,
                        "msg": msg,
                    }
                ).encode(),
            ]
        )

    def test_msg(self, to_, times=5, delay=1):
        for i in range(1, times + 1):
            self.send_msg(to_=to_, msg=f"checking are u okey - {i}", type_="test")
            time.sleep(delay)

    def _dealerListener(self):
        while True:
            try:
                identity, empty, msg = self.ctrl.recv_multipart(flags=zmq.NOBLOCK)
                if msg:
                    identity_str = identity.hex()
                    decoded_msg = json.loads(msg.decode())
                    formated_time = get_time()
                    mm = decoded_msg["msg"]
                    self.dealer_registry[decoded_msg["name"]] = decoded_msg[
                        "name"
                    ].encode()
                    # self.current_log[decoded_msg["name"]] = decoded_msg
                    self.log_registry[formated_time] = mm
                    # if decoded_msg["type"] in ("status", "log", "check"):
                    self.update_log_pane(
                        t=formated_time,
                        name=decoded_msg["name"],
                        type_=decoded_msg["type"],
                        log=mm,
                    )
                    # elif decoded_msg["type"] == "proc_out/err":
                    #     self.update_proc_output_pane(mm)
                    # else:
                    #     pass
                    self._executation_p(decoded_msg)
            except zmq.Again:
                pass
            except Exception as e:
                pass
            time.sleep(0.01)

    def _executation_p(self, decoded_msg):
        if (
            decoded_msg["name"] == "runner"
            and decoded_msg["type"] == "command"
            and decoded_msg["command"] == "control-system-shutdown"
        ):
            if self.application.is_running == True:
                self.application.exit()
        elif (
            decoded_msg["name"] == "runner"
            and decoded_msg["type"] == "command"
            and decoded_msg["command"] == "establish_connections"
        ):
            threading.Thread(
                target=self.send_msg,
                daemon=True,
                kwargs={
                    "to_": "all",
                    "type_": "command",
                    "msg": "establish_connections",
                    "command": "establish_connections",
                },
            ).start()
            # self.send_msg(to_="all", msg="establish_connections")
        elif (
            decoded_msg["name"] == "master_clock"
            and decoded_msg["type"] == "status"
            and decoded_msg["msg"] == "Tick Started"
        ):
            self.establish_connections()
            self.established_connections = True

    def send_msg(self, to_, msg, type_="status", **kwargs):
        if type(to_) == str:
            to_ = [
                to_,
            ]

        if to_[0] == "all":
            for id in self.dealer_registry.values():
                # self.update_log_pane(get_time(), self.name, "msg - send", f"sending msg to {id}")
                self.ctrl.send_multipart(
                    [
                        id,
                        b"",
                        json.dumps(
                            {
                                "type": type_,
                                "name": self.name,
                                "status": f"{'Running' if self.running else 'Not Running'}",
                                "msg": msg,
                                **kwargs,
                            }
                        ).encode(),
                    ]
                )
                if type_ == "test":
                    self.update_log_pane(get_time(), self.name, f"test - {id}", msg)
                time.sleep(0.5)
        elif type(to_) in (list, tuple, set):
            for id in [
                self.dealer_registry[i] for i in to_ if i in self.dealer_registry.keys()
            ]:
                self.ctrl.send_multipart(
                    [
                        id,
                        b"",
                        json.dumps(
                            {
                                "type": type_,
                                "name": self.name,
                                "status": f"{'Running' if self.running else 'Not Running'}",
                                "msg": msg,
                                **kwargs,
                            }
                        ).encode(),
                    ]
                )
                if type_ == "test":
                    self.update_log_pane(get_time(), self.name, f"test - {id}", msg)
                time.sleep(0.5)

    # def update_display_pane(self):
    #     self.display_pane.text = ""
    #     for dealer_name, dealer_msg in self.current_log.items():
    #         self.display_pane.buffer.insert_text(f" {dealer_name} - {dealer_msg}\n")

    def update_log_pane(self, t, name, type_, log):
        if t and log:
            self.log_pane.buffer.insert_text(
                f"{t} - {name:^30} - {type_:^15} - {log}\n"
            )

    def thread_spawner(self): ...

    def _window_start(self):
        self.application.run()

    def _tick_loop(self):
        while True:
            try:
                # self.update_log_pane(get_time(), self.name, "check", "tick loop check")
                data = self.tick.recv_json(flags=zmq.NOBLOCK)
                # self.current_log["master_clock"] = data
                self.log_registry[get_time()] = data
                # self.update_display_pane()
                # self.(get_time(), "master_clock", "tick", data)
                # try:
                #     data = self.tick.recv_json()
                #     self.update_log_pane(time, data)
                # except zmq.Again:
                #     # self.update_log_pane(time, "Error in the contorl system tick loop")
                #     pass
            except zmq.Again:
                pass
            time.sleep(1)

    def _heartbeat_loop(self): ...


if __name__ == "__main__":
    control_sys = ControlSystem()
    control_sys.start()
