"""
SynapseOS - Visualiser
======================
Text-based and matplotlib visualisations of the Adaptive Cell Fabric.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def print_network_ascii(net, max_cells: int = 20) -> None:
    """Simple textual overview of roles and connectivity density."""
    print("\n" + "═" * 62)
    print("  SYNAPSEOS – Adaptive Cell Fabric Snapshot")
    print("═" * 62)
    print(net.summary())
    print("-" * 62)

    # Show a sample of cells sorted by activity
    cells = sorted(net.cells.values(), key=lambda c: -c.activity)[:max_cells]
    print(f"{'ID':<8} {'Activity':>8}  Role mix")
    print("-" * 62)
    for c in cells:
        bar = "█" * int(c.activity * 10)
        print(f"{c.id:<8} {c.activity:>8.2f}  {c.get_role_summary()}  {bar}")

    print("-" * 62)
    print(f"Total connections : {net.connection_count()}")
    print(f"Average strength  : {net.average_strength():.3f}")
    print("═" * 62 + "\n")


def plot_fitness_history(history: List[Dict], title: str = "Evolution", save_path: Optional[str] = None):
    """Plot fitness, accuracy and energy over generations."""
    if not HAS_MPL:
        print("matplotlib not available – skipping plot")
        return

    gens = [h["generation"] for h in history]
    fitness = [h["fitness"] for h in history]
    acc = [h["accuracy"] for h in history]
    energy = [h["energy"] for h in history]
    links = [h["n_links"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].plot(gens, fitness, "b-", linewidth=2)
    axes[0, 0].set_title("Fitness")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gens, acc, "g-", linewidth=2)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(gens, energy, "r-", linewidth=2)
    axes[1, 0].set_title("Average Energy")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gens, links, "m-", linewidth=2)
    axes[1, 1].set_title("Number of Connections")
    axes[1, 1].set_xlabel("Generation")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_role_evolution(history: List[Dict], save_path: Optional[str] = None):
    """Stacked area chart of average role distribution over time."""
    if not HAS_MPL or not history:
        return

    gens = [h["generation"] for h in history]
    role_keys = list(history[0]["roles"].keys())
    data = {k: [h["roles"][k] for h in history] for k in role_keys}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(gens, *[data[k] for k in role_keys], labels=role_keys, alpha=0.85)
    ax.set_title("Average Role Distribution over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Saved role plot → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_connectivity_graph(net, title: str = "Cell Fabric", save_path: Optional[str] = None):
    """
    Spring-layout style graph of the current network.
    Node colour encodes dominant role.
    Edge width / alpha encodes strength.
    """
    if not HAS_MPL:
        print("matplotlib not available – skipping graph plot")
        return

    try:
        import networkx as nx
        HAS_NX = True
    except ImportError:
        HAS_NX = False

    if not HAS_NX:
        # Fallback: simple scatter of cells by id order
        fig, ax = plt.subplots(figsize=(8, 8))
        ids = list(net.cells.keys())
        n = len(ids)
        # Place on a circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {cid: (np.cos(a), np.sin(a)) for cid, a in zip(ids, angles)}

        # Draw edges
        for conn in net.connections:
            if conn.source_id in pos and conn.target_id in pos:
                x = [pos[conn.source_id][0], pos[conn.target_id][0]]
                y = [pos[conn.source_id][1], pos[conn.target_id][1]]
                ax.plot(x, y, color="gray", alpha=0.25 + 0.6 * conn.strength,
                        linewidth=0.5 + 2.0 * conn.strength)

        # Draw nodes coloured by dominant role
        role_colors = {
            "processing": "#e74c3c",
            "memory": "#3498db",
            "routing": "#2ecc71",
            "sensing": "#f39c12",
            "decision": "#9b59b6",
        }
        for cid, (x, y) in pos.items():
            cell = net.cells[cid]
            dom = max(cell.role, key=cell.role.get)
            ax.scatter(x, y, s=120 + 200 * cell.activity, c=role_colors.get(dom, "gray"),
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.text(x, y, cid[1:], fontsize=6, ha="center", va="center", color="white")

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        # Legend
        patches = [mpatches.Patch(color=c, label=k) for k, c in role_colors.items()]
        ax.legend(handles=patches, loc="upper left", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
            print(f"Saved graph → {save_path}")
        else:
            plt.show()
        plt.close()
        return

    # Full NetworkX version if available
    G = nx.DiGraph()
    for cid in net.cells:
        G.add_node(cid)
    for conn in net.connections:
        G.add_edge(conn.source_id, conn.target_id, weight=conn.strength)

    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(len(G)))

    fig, ax = plt.subplots(figsize=(9, 9))
    # Edges
    edges = G.edges(data=True)
    widths = [1 + 3 * d["weight"] for (_, _, d) in edges]
    alphas = [0.2 + 0.7 * d["weight"] for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, edge_color="gray",
                           arrows=True, arrowsize=8, ax=ax)

    # Nodes
    role_colors = {
        "processing": "#e74c3c",
        "memory": "#3498db",
        "routing": "#2ecc71",
        "sensing": "#f39c12",
        "decision": "#9b59b6",
    }
    node_colors = []
    node_sizes = []
    for cid in G.nodes:
        cell = net.cells[cid]
        dom = max(cell.role, key=cell.role.get)
        node_colors.append(role_colors.get(dom, "gray"))
        node_sizes.append(150 + 300 * cell.activity)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n[1:] for n in G.nodes},
                            font_size=6, font_color="white", ax=ax)

    ax.set_title(title)
    ax.axis("off")
    patches = [mpatches.Patch(color=c, label=k) for k, c in role_colors.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Saved graph → {save_path}")
    else:
        plt.show()
    plt.close()
