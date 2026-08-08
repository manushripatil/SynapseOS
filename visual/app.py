"""
SynapseOS Visual Simulator
==========================
Live web UI for the Adaptive Cell Fabric.

Optimised for free-tier hosts (Render): cached snapshots, lighter steps,
non-blocking state reads so buttons stay responsive.

Run locally:
    python3 visual/app.py
"""

from __future__ import annotations
import sys
import os
import time
import threading
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
import numpy as np

from network import Network
from evolution import EvolutionEngine, anomaly_detection_task, pattern_recognition_task
from simulator import Simulator

_VISUAL_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    static_folder=os.path.join(_VISUAL_DIR, "static"),
    template_folder=os.path.join(_VISUAL_DIR, "templates"),
)

# Lighter tasks for the web UI (fewer samples = faster button responses)
def _web_anomaly(net: Network):
    return anomaly_detection_task(net, n_samples=12)


def _web_pattern(net: Network):
    return pattern_recognition_task(net, n_samples=10)


class LiveSim:
    """Thread-safe live simulation with a cached snapshot for instant reads."""

    def __init__(self):
        self._lock = threading.Lock()
        self.sim: Optional[Simulator] = None
        self.running = False
        self.task_name = "anomaly"
        self.generation = 0
        self.last_accuracy = 0.0
        self.last_energy = 0.0
        self.last_fitness = 0.0
        self.message = "Ready"
        self.history: list = []
        self._thread: Optional[threading.Thread] = None
        # Slower default = less CPU on free tier
        self.speed = 0.45
        self._cache: Dict[str, Any] = {}
        self.reset(n_cells=20, seed=7)

    def _task(self) -> Callable:
        if self.task_name == "pattern":
            return _web_pattern
        return _web_anomaly

    def _rebuild_cache(self) -> Dict[str, Any]:
        """Build snapshot. Caller must hold lock OR be sole mutator."""
        if self.sim is None:
            self._cache = {}
            return self._cache
        net = self.sim.net
        cells = []
        maturities = []
        for cid, cell in net.cells.items():
            dom = max(cell.role, key=cell.role.get)
            maturities.append(cell.maturity)
            cells.append({
                "id": cid,
                "role": {k: round(v, 3) for k, v in cell.role.items()},
                "dominant": dom,
                "activity": round(cell.activity, 3),
                "maturity": round(cell.maturity, 3),
                "age": cell.age,
            })
        avg_maturity = float(sum(maturities) / len(maturities)) if maturities else 0.0
        links = [{
            "source": c.source_id,
            "target": c.target_id,
            "strength": round(c.strength, 3),
        } for c in net.connections]
        roles = net.role_distribution()
        self._cache = {
            "generation": self.generation,
            "n_cells": len(net.cells),
            "n_links": len(net.connections),
            "accuracy": round(self.last_accuracy, 3),
            "energy": round(self.last_energy, 3),
            "fitness": round(self.last_fitness, 3),
            "avg_strength": round(net.average_strength(), 3),
            "roles": {k: round(v, 3) for k, v in roles.items()},
            "task": self.task_name,
            "running": self.running,
            "message": self.message,
            "cells": cells,
            "links": links,
            "history": self.history[-40:],
            "avg_maturity": round(avg_maturity, 3),
            "energy_budget": round(net.energy_budget, 2),
            "energy_spent": round(getattr(net, "energy_spent", 0.0), 2),
        }
        return self._cache

    def snapshot(self) -> Dict[str, Any]:
        """Instant: return last cache (no lock wait on hot path)."""
        cache = self._cache
        if not cache:
            with self._lock:
                return dict(self._rebuild_cache())
        # Running flag can change without full rebuild
        out = dict(cache)
        out["running"] = self.running
        out["message"] = self.message
        return out

    def reset(self, n_cells: int = 20, seed: int = 7):
        self.running = False
        with self._lock:
            self.sim = Simulator(n_cells=n_cells, seed=seed)
            self.generation = 0
            self.last_accuracy = 0.0
            self.last_energy = 0.0
            self.last_fitness = 0.0
            self.history = []
            self.message = f"Reset — {n_cells} cells"
            self._rebuild_cache()

    def step_once(self):
        with self._lock:
            if self.sim is None:
                return
            task = self._task()
            result = self.sim.evo.step(
                task,
                do_structural_mutation=(self.generation % 8 == 0),
                fast=True,
            )
            self.generation = self.sim.net.generation
            rec = self.sim.evo.history[-1]
            self.last_accuracy = rec["accuracy"]
            self.last_energy = rec["energy"]
            self.last_fitness = rec["fitness"]
            self.history.append({
                "gen": self.generation,
                "acc": self.last_accuracy,
                "fit": self.last_fitness,
                "cells": len(self.sim.net.cells),
            })
            if len(self.history) > 100:
                self.history = self.history[-100:]
            pop = result.get("pop_stats", {})
            born = pop.get("born", 0)
            died = pop.get("died", 0)
            extra = f" | +{born}/-{died}" if (born or died) else ""
            self.message = (
                f"Gen {self.generation} | acc={self.last_accuracy:.2f} | "
                f"cells={len(self.sim.net.cells)}{extra}"
            )
            self._rebuild_cache()

    def damage(self, n: int = 2):
        with self._lock:
            if self.sim is None:
                return []
            victims = self.sim.inject_damage(n, verbose=False)
            self.message = f"Damaged {len(victims)} cells"
            self._rebuild_cache()
            return victims

    def set_task(self, name: str):
        self.task_name = name
        self.message = f"Task → {name}"
        # cheap cache touch
        if self._cache:
            self._cache = dict(self._cache)
            self._cache["task"] = name
            self._cache["message"] = self.message

    def start(self):
        if self.running:
            return
        self.running = True
        self.message = "Running…"
        if self._cache:
            self._cache = dict(self._cache)
            self._cache["running"] = True
            self._cache["message"] = self.message
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.message = "Paused"
        if self._cache:
            self._cache = dict(self._cache)
            self._cache["running"] = False
            self._cache["message"] = self.message

    def _loop(self):
        while self.running:
            t0 = time.time()
            try:
                self.step_once()
            except Exception as e:
                self.message = f"Error: {e}"
                self.running = False
                break
            # Sleep remaining time so speed setting is respected
            elapsed = time.time() - t0
            time.sleep(max(0.05, self.speed - elapsed))


live = LiveSim()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    # Instant response — no waiting on evolution lock
    return jsonify(live.snapshot())


@app.route("/api/start", methods=["POST"])
def api_start():
    live.start()
    return jsonify({"ok": True, "message": live.message, "running": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    live.stop()
    return jsonify({"ok": True, "message": live.message, "running": False})


@app.route("/api/step", methods=["POST"])
def api_step():
    live.step_once()
    return jsonify(live.snapshot())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n_cells", 20))
    n = max(12, min(n, 40))  # cap for free tier
    live.reset(n_cells=n)
    return jsonify(live.snapshot())


@app.route("/api/damage", methods=["POST"])
def api_damage():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 2))
    victims = live.damage(n)
    snap = live.snapshot()
    snap["victims"] = victims
    return jsonify(snap)


@app.route("/api/task", methods=["POST"])
def api_task():
    data = request.get_json(silent=True) or {}
    live.set_task(data.get("task", "anomaly"))
    return jsonify({"ok": True, "task": live.task_name, "message": live.message})


@app.route("/api/speed", methods=["POST"])
def api_speed():
    data = request.get_json(silent=True) or {}
    live.speed = float(data.get("speed", 0.45))
    return jsonify({"ok": True, "speed": live.speed})


@app.route("/api/killer_demo", methods=["POST"])
def api_killer_demo():
    """Damage a fraction of cells (recovery continues via Start / already running)."""
    data = request.get_json(silent=True) or {}
    frac = float(data.get("frac", 0.30))
    with live._lock:
        if live.sim is None:
            return jsonify({"ok": False})
        n = max(2, int(len(live.sim.net.cells) * frac))
        pre = live.last_accuracy
        victims = live.sim.inject_damage(n, verbose=False)
        live.message = f"KILLER DEMO: destroyed {len(victims)} cells ({frac:.0%})"
        # One light recovery step only (rest via live evolution)
        try:
            live.sim.evo.step(live._task(), recovery_mode=True, fast=True)
            live.generation = live.sim.net.generation
            if live.sim.evo.history:
                rec = live.sim.evo.history[-1]
                live.last_accuracy = rec["accuracy"]
                live.last_energy = rec["energy"]
                live.last_fitness = rec["fitness"]
        except Exception:
            pass
        live._rebuild_cache()
        snap = dict(live._cache)
    snap["victims"] = victims
    snap["pre_accuracy"] = pre
    return jsonify(snap)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n  SynapseOS Visual Simulator")
    print(f"  -> http://127.0.0.1:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
