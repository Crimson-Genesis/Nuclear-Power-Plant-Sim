import zmq

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("ipc:///tmp/clock.tick")
sub.setsockopt_string(zmq.SUBSCRIBE, "")  # all topics
while True:
    msg = sub.recv_json()
    # msg["type"] == "tick" or "status"
    if msg["type"] == "tick":
        sim_time = msg["sim_time"]
        print(msg)
        # perform one simulation step now
