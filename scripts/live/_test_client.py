#!/usr/bin/env python
"""Synthetic smoke client for planner_server.py (no Isaac). Sends N random POV
frames + a just_reset mask, prints the returned action. Validates the full
request->plan->response path + protocol on the host venv alone."""
import pickle
import socket
import struct
import sys
import time

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5557


def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b.extend(c)
    return bytes(b)


def send_msg(s, o):
    d = pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack(">I", len(d)) + d)


def recv_msg(s):
    (n,) = struct.unpack(">I", recv_exact(s, 4))
    return pickle.loads(recv_exact(s, n))


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", PORT))
print(f"connected to 127.0.0.1:{PORT}, sending {N} envs")

for step, reset in enumerate([True, False, False]):
    req = {
        "just_reset": np.array([reset] * N, dtype=bool),
        "pix": (np.random.rand(N, 224, 224, 3) * 255).astype(np.uint8),
        "state": np.zeros((N, 21), dtype=np.float32),
    }
    t = time.time()
    send_msg(s, req)
    resp = recv_msg(s)
    dt = time.time() - t
    a = np.asarray(resp["action"])  # server sends .tolist(); works for list or array
    ok = bool((a >= -1).all() and (a <= 1).all())
    extra = ""
    if "flee" in resp:                       # detector mode
        extra += f" flee={resp['flee']}"
    if "wm_danger" in resp:                  # planner mode -- showcase signals
        wd = np.asarray(resp["wm_danger"])
        extra += f" wm_danger[med={np.median(wd):.2f} max={wd.max():.2f}]"
    if "det_logit" in resp:                  # planner or detector mode
        dl = np.asarray(resp["det_logit"])
        extra += f" det_logit[med={np.median(dl):.2f} max={dl.max():.2f}]"
    print(f"step {step} reset={reset} shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] "
          f"ok={ok} {dt:.3f}s  a0={np.round(a[0],3)}{extra}")
s.close()
print("done")
