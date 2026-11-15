"""Part 3: VCG (Clarke pivot) mechanism for buying the cheapest A -> F path.

Every edge of the network (Figure 7.3 of the textbook) is a selfish agent
whose private cost is incurred only if the edge is used. The buyer asks every
edge to report its cost, buys the cheapest A -> F path under the reports, and
pays each edge e

    p_e = C(G without e) - [C(G) - c_e]   if e is on the chosen path,
    p_e = C(G without e) - C(G) = 0      otherwise,

where C(.) is the cost of the cheapest path under the reported costs. An
edge's utility is p_e - (true cost of e if it is used, else 0).

The script reproduces the payments for truthful reports, then sweeps the
reported cost of one on-path edge and one off-path edge from 0 to 10 in steps
of 0.5 and records how that edge's utility changes. It also repeats the
sweep for every edge as a robustness check.
"""

import numpy as np

from common import save_fig, save_json, plt

NODES = ["A", "B", "C", "D", "E", "F"]
SOURCE, TARGET = "A", "F"

# true (private) costs of the edges in Figure 7.3; the graph is undirected
TRUE_COSTS = {
    ("A", "B"): 3,
    ("A", "C"): 2,
    ("B", "D"): 2,
    ("B", "E"): 1,
    ("C", "E"): 3,
    ("C", "F"): 5,
    ("D", "F"): 2,
    ("E", "F"): 1,
}
EDGES = list(TRUE_COSTS)
REPORT_GRID = np.round(np.arange(0, 10.0001, 0.5), 1)
ON_PATH_EDGE = ("A", "B")
OFF_PATH_EDGE = ("C", "E")


def name(edge):
    return "".join(edge)


# --------------------------------------------------------------------------
# Cheapest path
# --------------------------------------------------------------------------
def simple_paths(edges, source=SOURCE, target=TARGET):
    """All simple source -> target paths, each as a tuple of nodes."""
    adj = {n: [] for n in NODES}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    out, stack = [], [(source, (source,))]
    while stack:
        node, path = stack.pop()
        if node == target:
            out.append(path)
            continue
        for nxt in adj[node]:
            if nxt not in path:
                stack.append((nxt, path + (nxt,)))
    return out


def path_edges(path):
    return [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]


def cheapest_path(costs, excluded=None):
    """Cheapest A -> F path under `costs`, optionally without edge `excluded`.

    Ties are broken deterministically: fewer edges first, then the
    alphabetical order of the node sequence.
    """
    edges = [e for e in costs if e != excluded]
    best = None
    for p in simple_paths(edges):
        c = sum(costs[e] for e in path_edges(p))
        key = (c, len(p), "".join(p))
        if best is None or key < best[0]:
            best = (key, p)
    if best is None:
        return float("inf"), None
    return best[0][0], best[1]


# --------------------------------------------------------------------------
# VCG
# --------------------------------------------------------------------------
def vcg(reported):
    """Chosen path, its reported cost and the VCG payment to every edge."""
    total, path = cheapest_path(reported)
    on_path = set(path_edges(path))
    payments = {}
    for e in reported:
        cost_without_e, _ = cheapest_path(reported, excluded=e)
        others_cost_on_path = total - (reported[e] if e in on_path else 0)
        payments[e] = cost_without_e - others_cost_on_path
    return path, total, payments


def utility(edge, reported, true_costs=TRUE_COSTS):
    path, _, pay = vcg(reported)
    used = edge in set(path_edges(path))
    return pay[edge] - (true_costs[edge] if used else 0), used, pay[edge], path


def sweep(edge):
    rows = []
    for x in REPORT_GRID:
        reported = dict(TRUE_COSTS)
        reported[edge] = float(x)
        u, used, pay, path = utility(edge, reported)
        rows.append({"reported_cost": float(x), "on_path": used, "payment": float(pay),
                     "utility": float(u), "chosen_path": "".join(path)})
    return rows


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
POS = {"A": (0.0, 1.0), "B": (1.3, 1.9), "C": (1.3, 0.1), "D": (2.7, 2.2),
       "E": (2.5, 1.05), "F": (3.9, 1.0)}


def plot_graph(path):
    chosen = set(path_edges(path))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for e, c in TRUE_COSTS.items():
        (x1, y1), (x2, y2) = POS[e[0]], POS[e[1]]
        on = e in chosen
        ax.plot([x1, x2], [y1, y2], color="#c0392b" if on else "#7f8c8d", lw=3 if on else 1.4, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(c), ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=3)
    for n, (x, y) in POS.items():
        ax.scatter([x], [y], s=420, color="white", edgecolor="black", zorder=2)
        ax.text(x, y, n, ha="center", va="center", fontsize=10, weight="bold", zorder=4)
    ax.set_xlim(-0.35, 4.25)
    ax.set_ylim(-0.25, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Network of Figure 7.3 with true edge costs; cheapest path A–B–E–F in red")
    save_fig(fig, "part3_graph.png")


def plot_sweeps(sweeps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, edge in zip(axes, (ON_PATH_EDGE, OFF_PATH_EDGE)):
        rows = sweeps[name(edge)]
        x = [r["reported_cost"] for r in rows]
        ax.step(x, [r["utility"] for r in rows], where="mid", color="#1f5aa6", lw=1.8, label="Utility of the edge")
        ax.plot(x, [r["utility"] for r in rows], "o", ms=3, color="#1f5aa6")
        ax.step(x, [r["payment"] for r in rows], where="mid", color="#7f8c8d", lw=1, ls="--", label="VCG payment")
        ax.axvline(TRUE_COSTS[edge], color="#c0392b", ls=":", lw=1.2, label="True cost")
        ax.axhline(0, color="grey", lw=0.6)
        where = "on" if edge == ON_PATH_EDGE else "off"
        ax.set_title(f"Edge {name(edge)} ({where} the chosen path, true cost {TRUE_COSTS[edge]})")
        ax.set_xlabel("Reported cost")
        ax.set_xticks(range(0, 11))
    axes[0].set_ylabel("Payment / utility")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "part3_misreport_sweep.png")


def main():
    path, total, payments = vcg(TRUE_COSTS)
    print(f"Chosen path: {'-'.join(path)}  (cost {total})")
    truthful = []
    for e in EDGES:
        cost_wo, path_wo = cheapest_path(TRUE_COSTS, excluded=e)
        u, used, pay, _ = utility(e, TRUE_COSTS)
        truthful.append({"edge": name(e), "true_cost": TRUE_COSTS[e], "on_path": used,
                         "cheapest_without_edge": cost_wo, "path_without_edge": "".join(path_wo),
                         "payment": pay, "utility": u})
        print(f"  {name(e)}: cost {TRUE_COSTS[e]}, on path={used!s:5}, C(G-e)={cost_wo} via "
              f"{'-'.join(path_wo)}, payment={pay}, utility={u}")
    print(f"  total paid = {sum(payments.values())}, true cost of path = {total}")

    sweeps = {name(e): sweep(e) for e in EDGES}
    for e in (ON_PATH_EDGE, OFF_PATH_EDGE):
        print(f"Sweep for edge {name(e)} (true cost {TRUE_COSTS[e]}):")
        for r in sweeps[name(e)]:
            print(f"  report {r['reported_cost']:4.1f}: path {r['chosen_path']}, on path={r['on_path']!s:5}, "
                  f"payment={r['payment']:4.1f}, utility={r['utility']:+.1f}")

    robustness = []
    for e in EDGES:
        u_true = next(r["utility"] for r in sweeps[name(e)] if r["reported_cost"] == TRUE_COSTS[e])
        best = max(sweeps[name(e)], key=lambda r: r["utility"])
        robustness.append({"edge": name(e), "utility_truthful": u_true, "best_utility_any_report": best["utility"],
                           "max_gain_from_lying": best["utility"] - u_true})
        print(f"  {name(e)}: truthful utility {u_true:+.1f}, best over all reports {best['utility']:+.1f}, "
              f"gain from lying {best['utility'] - u_true:+.1f}")

    plot_graph(path)
    plot_sweeps(sweeps)
    save_json({"chosen_path": "".join(path), "path_cost": total, "truthful": truthful,
               "total_payment": sum(payments.values()), "sweeps": sweeps,
               "robustness": robustness}, "part3_results.json")


if __name__ == "__main__":
    main()
