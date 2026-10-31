"""Live knee-load estimate: run the GRF-free surrogate on the estimator's output.  [needs torch]

The surrogate takes a 20-frame window at 100 Hz centred on the query frame. Online, the last 0.2 s of estimator
output is resampled to that grid, so each prediction refers to the frame 0.095 s in the past (reported as
surrogate_time). Body positions come from the generic URDF model scaled to the subject's height, not a scaled
OpenSim model, so the features are approximate. Each foot is the checkpoint's foot_point on its calcn body (the
calcn origin unless the model was trained on another point).

Contact flags in the buffer can be revised after the fact: state["contact_onset"] = {"r": t} (the detector's
back-dated heel strike) sets that foot's flag from t on, so the window the network sees starts stance at heel
strike rather than at foot-flat. The flags used at the query frame are returned as contact_used.
"""
from collections import deque

import numpy as np

WINDOW, FS = 20, 100.0


class LiveSurrogate:
    def __init__(self, checkpoint, mass_kg, height_m):
        import torch
        from inference_posture_surrogate import JCFPostureSurrogate
        torch.set_num_threads(1)
        self.m = JCFPostureSurrogate(checkpoint, device="cpu")
        self.mass, self.height = mass_kg, height_m
        self.buf = deque()

    def update(self, state):
        p = state["pelvis"]
        c = state["contact"]
        fp = self.m.foot_point
        foot = {s: state[f"calcn_{s}"] + (state[f"calcn_{s}_R"] @ fp if fp.any() else 0.0) for s in ("r", "l")}
        self.buf.append([state["t"], np.asarray(state["q"][:16]), state["com"] - p, foot["r"] - p,
                         foot["l"] - p, float(c["r"]), float(c["l"])])
        for side, t_on in (state.get("contact_onset") or {}).items():
            i = 5 if side == "r" else 6
            for b in self.buf:
                if b[0] >= t_on:
                    b[i] = 1.0
        t_end = state["t"]
        while len(self.buf) > 2 and self.buf[1][0] < t_end - (WINDOW + 2) / FS:
            self.buf.popleft()
        if self.buf[0][0] > t_end - (WINDOW - 1) / FS:
            return None
        tg = t_end - np.arange(WINDOW)[::-1] / FS
        ts = np.array([b[0] for b in self.buf])
        col = lambda i: np.column_stack([np.interp(tg, ts, np.array([b[i] for b in self.buf])[:, j])
                                         for j in range(np.array(self.buf[0][i]).size)])
        q = col(1)[WINDOW // 2]
        flags = lambda i: (np.interp(tg, ts, [b[i] for b in self.buf]) > 0.5).astype(float)
        fr, fl = flags(5), flags(6)
        kw = dict(r_contact_window=fr, l_contact_window=fl) if self.m.include_contact else {}
        out = self.m.predict_windowed(q, col(2), col(3), col(4), mass_kg=self.mass, height_m=self.height, **kw)
        g = lambda k: float(np.ravel(out[k])[0])
        return dict(f_medial=g("f_medial"), fy=g("fy"), mx=g("mx"), surrogate_time=float(tg[WINDOW // 2]),
                    contact_used={"r": bool(fr[WINDOW // 2]), "l": bool(fl[WINDOW // 2])})
