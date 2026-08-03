"""
SynapseOS Visual Simulator — Phase 1
====================================
Live web UI for the Adaptive Cell Fabric.

Run:
    cd synapseos
    python3 visual/app.py

Then open http://127.0.0.1:5050
"""

from __future__ import annotations
import sys
import os
import time
import threading
from typing import Dict, Any, Optional

# Allow imports from parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
import numpy as np

from network import Network
from evolution import EvolutionEngine, anomaly_detection_task, pattern_recognition_task
from simulator import Simulator

app = Flask(__name__, static_folder="static", template_folder="templates")

# ------------------------------------------------------------------
# Simulation state (shared)
# ------------------------------------------------------------------
class LiveSim:
    def __init__(self):
        self.lock = threading.Lock()
        self.sim: Optional[Simulator] = None
        self.running = False
        self.task_name = "anomaly"
        self.generation = 0
        self.last_accuracy = 0.0
        self.last_energy = 0.0
        self.last_fitness = 0.0
        self.message = "Ready"
        self.history = []
        self._thread: Optional[threading.Thread] = None
        self.speed = 0.15  # seconds between generations
        self.reset(n_cells=36, seed=7)

    def reset(self, n_cells: int = 36, seed: int = 7):
        with self.lock:
            self.sim = Simulator(n_cells=n_cells, seed=seed)
            self.generation = 0
            self.last_accuracy = 0.0
            self.last_energy = 0.0
            self.last_fitness = 0.0
            self.history = []
            self.message = f"Reset — {n_cells} cells"
            self.running = False

    def _task(self):
        if self.task_name == "pattern":
            return pattern_recognition_task
        return anomaly_detection_task

    def step_once(self):
        with self.lock:
            if self.sim is None:
                return
            task = self._task()
            result = self.sim.evo.step(task, do_structural_mutation=(self.generation % 6 == 0), fast=True)
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
            if len(self.history) > 200:
                self.history = self.history[-200:]
            pop = result.get("pop_stats", {})
            born = pop.get("born", 0)
            died = pop.get("died", 0)
            extra = ""
            if born or died:
                extra = f" | +{born}/-{died} cells"
            self.message = f"Gen {self.generation} | acc={self.last_accuracy:.2f} | cells={len(self.sim.net.cells)}{extra}"

    def damage(self, n: int = 3):
        with self.lock:
            if self.sim is None:
                return []
            victims = self.sim.inject_damage(n, verbose=False)
            self.message = f"Damaged {len(victims)} cells: {', '.join(victims)}"
            return victims

    def set_task(self, name: str):
        with self.lock:
            self.task_name = name
            self.message = f"Task → {name}"

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            if self.sim is None:
                return {}
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
            links = []
            for c in net.connections:
                links.append({
                    "source": c.source_id,
                    "target": c.target_id,
                    "strength": round(c.strength, 3),
                })
            roles = net.role_distribution()
            return {
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
                "history": self.history[-60:],
                "avg_maturity": round(avg_maturity, 3),
                "energy_budget": round(net.energy_budget, 2),
                "energy_spent": round(net.energy_spent, 2),
            }

    def _loop(self):
        while self.running:
            self.step_once()
            time.sleep(self.speed)

    def start(self):
        if self.running:
            return
        self.running = True
        self.message = "Running…"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.message = "Paused"


live = LiveSim()


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(live.snapshot())


@app.route("/api/start", methods=["POST"])
def api_start():
    live.start()
    return jsonify({"ok": True, "message": live.message})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    live.stop()
    return jsonify({"ok": True, "message": live.message})


@app.route("/api/step", methods=["POST"])
def api_step():
    live.step_once()
    return jsonify(live.snapshot())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n_cells", 36))
    live.reset(n_cells=n)
    return jsonify(live.snapshot())


@app.route("/api/damage", methods=["POST"])
def api_damage():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 3))
    victims = live.damage(n)
    snap = live.snapshot()
    snap["victims"] = victims
    return jsonify(snap)


@app.route("/api/task", methods=["POST"])
def api_task():
    data = request.get_json(silent=True) or {}
    name = data.get("task", "anomaly")
    live.set_task(name)
    return jsonify({"ok": True, "task": live.task_name, "message": live.message})


@app.route("/api/speed", methods=["POST"])
def api_speed():
    data = request.get_json(silent=True) or {}
    live.speed = float(data.get("speed", 0.15))
    return jsonify({"ok": True, "speed": live.speed})



@app.route("/api/killer_demo", methods=["POST"])
def api_killer_demo():
    """Run a dramatic damage burst then continue evolution in recovery mode."""
    data = request.get_json(silent=True) or {}
    frac = float(data.get("frac", 0.35))
    with live.lock:
        if live.sim is None:
            return jsonify({"ok": False})
        n = max(2, int(len(live.sim.net.cells) * frac))
        # snapshot pre accuracy roughly from last
        pre = live.last_accuracy
        victims = live.sim.inject_damage(n, verbose=False)
        live.message = f"KILLER DEMO: destroyed {len(victims)} cells ({frac:.0%})"
        # force a few recovery steps inline
        for _ in range(3):
            live.sim.evo.step(
                live._task(),
                recovery_mode=True,
                fast=True,
            )
        live.generation = live.sim.net.generation
        if live.sim.evo.history:
            rec = live.sim.evo.history[-1]
            live.last_accuracy = rec["accuracy"]
            live.last_energy = rec["energy"]
            live.last_fitness = rec["fitness"]
        snap = live.snapshot()
        snap["victims"] = victims
        snap["pre_accuracy"] = pre
        return jsonify(snap)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n  SynapseOS Visual Simulator")
    print(f"  -> http://127.0.0.1:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
