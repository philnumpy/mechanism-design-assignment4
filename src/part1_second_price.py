"""Part 1: second-price sealed-bid (Vickrey) auction.

Bidder 1 has a fixed private value v in {0.2, 0.5, 0.8}. Each rival draws a
value from U[0, 1] and bids according to one of three strategies:

    truthful : bid = value
    half     : bid = value / 2
    random   : bid ~ U[0, 1], independent of the value

The highest bid wins and pays the highest competing bid, so bidder 1's utility
is v - max(rival bids) on a win and 0 on a loss.

Experiments
    A : sweep bidder 1's bid b = 0.00, 0.01, ..., 1.00 against random rivals
        (one rival -> 9 plots, two rivals -> 9 plots).
    B : fix the rival bid(s) and verify that b = v is a best response.
    C : A and B repeated with two rivals (bidders 2 and 3) using the same
        strategy.
"""

import argparse

import numpy as np

from common import SEED, save_fig, save_json, plt

VALUES = [0.2, 0.5, 0.8]
STRATEGIES = ["truthful", "half", "random"]
STRATEGY_LABEL = {
    "truthful": "Rival bids truthfully",
    "half": "Rival bids half its value",
    "random": "Rival bids U[0,1] at random",
}
BIDS = np.round(np.arange(0, 101) / 100.0, 2)  # 0.00, 0.01, ..., 1.00
FIXED_RIVAL_BIDS = [0.1, 0.4, 0.9]
TIE_TOL = 1e-9  # bids closer than this are treated as equal


# --------------------------------------------------------------------------
# Rival behaviour
# --------------------------------------------------------------------------
def rival_bids(strategy, n_auctions, n_rivals, rng):
    """Bids of `n_rivals` rivals in each of `n_auctions` auctions."""
    values = rng.uniform(0.0, 1.0, size=(n_auctions, n_rivals))
    if strategy == "truthful":
        return values
    if strategy == "half":
        return 0.5 * values
    if strategy == "random":
        return rng.uniform(0.0, 1.0, size=(n_auctions, n_rivals))
    raise ValueError(f"unknown strategy: {strategy}")


# --------------------------------------------------------------------------
# Auction outcome for bidder 1
# --------------------------------------------------------------------------
def utility_bidder1(v, b, bids_rivals, rng=None):
    """Utility of bidder 1 (value v, bid b) in each auction.

    bids_rivals has shape (n_auctions, n_rivals). Ties for the highest bid
    are broken uniformly at random among the tied bidders; with
    `rng=None` the expected utility of the lottery is returned instead.
    """
    bids_rivals = np.atleast_2d(bids_rivals)
    top = bids_rivals.max(axis=1)
    n_tied_rivals = (np.abs(bids_rivals - top[:, None]) <= TIE_TOL).sum(axis=1)

    win = b > top + TIE_TOL
    tie = np.abs(b - top) <= TIE_TOL
    payoff = v - top  # pays the highest competing bid

    share = np.where(win, 1.0, 0.0)
    tie_share = 1.0 / (n_tied_rivals + 1)
    if rng is None:
        share = np.where(tie, tie_share, share)
    else:
        share = np.where(tie, (rng.uniform(size=top.shape) < tie_share), share)
    return share * payoff + 0.0  # + 0.0 turns -0.0 into 0.0


def expected_utility_curve(v, strategy, n_rivals, n_auctions, rng, common_draws=True):
    """Average utility of bidder 1 for every bid in BIDS.

    With common_draws=True each bid faces the same n_auctions rival
    realisations (common random numbers); otherwise every bid gets fresh draws.
    """
    means = np.empty(len(BIDS))
    ses = np.empty(len(BIDS))
    shared = rival_bids(strategy, n_auctions, n_rivals, rng) if common_draws else None
    for i, b in enumerate(BIDS):
        draws = shared if common_draws else rival_bids(strategy, n_auctions, n_rivals, rng)
        u = utility_bidder1(v, b, draws)  # ties have probability 0 here
        means[i] = u.mean()
        ses[i] = u.std(ddof=1) / np.sqrt(n_auctions)
    return means, ses


def analytic_utility(v, b, strategy, n_rivals):
    """Closed-form E[u(b)] = integral_0^b (v - r) f(r) dr, r = max rival bid."""
    b = np.asarray(b, dtype=float)
    if strategy in ("truthful", "random"):  # r_j ~ U[0, 1]
        x = np.clip(b, 0, 1)
        if n_rivals == 1:
            return v * x - x**2 / 2
        return v * x**2 - 2 * x**3 / 3
    if strategy == "half":  # r_j ~ U[0, 0.5]
        x = np.clip(b, 0, 0.5)
        if n_rivals == 1:
            return 2 * v * x - x**2
        return 4 * v * x**2 - 8 * x**3 / 3
    raise ValueError(strategy)


# --------------------------------------------------------------------------
# Experiment A (and its two-rival repeat)
# --------------------------------------------------------------------------
def experiment_a(n_rivals, n_auctions, rng, common_draws=True):
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2), sharex=True)
    for r, v in enumerate(VALUES):
        for c, s in enumerate(STRATEGIES):
            means, ses = expected_utility_curve(v, s, n_rivals, n_auctions, rng, common_draws)
            exact = analytic_utility(v, BIDS, s, n_rivals)
            i_best = int(np.argmax(means))
            i_truth = int(np.argmin(np.abs(BIDS - v)))
            # every bid whose simulated utility ties the maximum
            best_set = BIDS[np.abs(means - means[i_best]) <= 1e-12]
            results.append(
                {
                    "n_rivals": n_rivals,
                    "value": v,
                    "strategy": s,
                    "best_bid_simulated": float(BIDS[i_best]),
                    "best_bid_set_min": float(best_set.min()),
                    "best_bid_set_max": float(best_set.max()),
                    "utility_at_truthful_bid": float(means[i_truth]),
                    "max_utility_simulated": float(means[i_best]),
                    "se_at_truthful_bid": float(ses[i_truth]),
                    "exact_utility_at_truthful_bid": float(exact[i_truth]),
                    "max_abs_error_vs_exact": float(np.max(np.abs(means - exact))),
                    "truthful_is_optimal": bool(means[i_truth] >= means[i_best] - 1e-12),
                    "curve": {"bids": BIDS.tolist(), "mean": means.tolist(), "se": ses.tolist()},
                }
            )

            ax = axes[r, c]
            ax.plot(BIDS, means, color="#1f5aa6", lw=1.6, label="Simulated mean")
            ax.fill_between(BIDS, means - 1.96 * ses, means + 1.96 * ses, color="#1f5aa6", alpha=0.2, lw=0)
            ax.plot(BIDS, exact, color="black", lw=0.9, ls="--", label="Closed form")
            ax.axvline(v, color="#c0392b", lw=1, ls=":", label="b = v")
            ax.plot(BIDS[i_truth], means[i_truth], "o", color="#c0392b", ms=4)
            ax.axhline(0, color="grey", lw=0.6)
            if r == 0:
                ax.set_title(STRATEGY_LABEL[s])
            if c == 0:
                ax.set_ylabel(f"v = {v}\nAverage utility of bidder 1")
            if r == 2:
                ax.set_xlabel("Bid of bidder 1, b")
            ax.set_xlim(0, 1)

            # individual plot as well (the assignment asks for 9 plots)
            f1, a1 = plt.subplots(figsize=(4.2, 3.0))
            a1.plot(BIDS, means, color="#1f5aa6", lw=1.6, label="Simulated mean")
            a1.plot(BIDS, exact, color="black", lw=0.9, ls="--", label="Closed form")
            a1.axvline(v, color="#c0392b", lw=1, ls=":", label="b = v")
            a1.axhline(0, color="grey", lw=0.6)
            a1.set_xlabel("Bid of bidder 1, b")
            a1.set_ylabel("Average utility of bidder 1")
            a1.set_title(f"v = {v}, {STRATEGY_LABEL[s].lower()}, {n_rivals} rival(s)")
            a1.legend(loc="lower left")
            save_fig(f1, f"part1A_{n_rivals}rival_v{v}_{s}.png")

    axes[0, 0].legend(loc="lower left")
    fig.suptitle(
        f"Second-price auction: bidder 1's average utility vs. bid "
        f"({n_rivals} rival{'s' if n_rivals > 1 else ''}, {n_auctions:,} auctions per bid)",
        y=1.0,
    )
    fig.tight_layout()
    save_fig(fig, f"part1A_grid_{n_rivals}rival.png")
    return results


# --------------------------------------------------------------------------
# Experiment B (and its two-rival repeat)
# --------------------------------------------------------------------------
def experiment_b(n_rivals):
    """Deterministic rival bids; utilities are exact (ties split evenly)."""
    if n_rivals == 1:
        profiles = [(r,) for r in FIXED_RIVAL_BIDS]
    else:
        profiles = [(r2, r3) for r2 in FIXED_RIVAL_BIDS for r3 in FIXED_RIVAL_BIDS]

    rows = []
    for v in VALUES:
        for prof in profiles:
            u = np.array([utility_bidder1(v, b, np.array([prof]))[0] for b in BIDS])
            u_truth = u[int(np.argmin(np.abs(BIDS - v)))]
            best = u.max()
            best_bids = BIDS[np.abs(u - best) <= 1e-12]
            rows.append(
                {
                    "n_rivals": n_rivals,
                    "value": v,
                    "rival_bids": list(prof),
                    "utility_truthful": round(float(u_truth), 6),
                    "max_utility": round(float(best), 6),
                    "optimal_bid_range": [float(best_bids.min()), float(best_bids.max())],
                    "truthful_is_optimal": bool(u_truth >= best - 1e-12),
                    "curve": u.tolist(),
                }
            )

    # plot: rows = v, columns = three rival profiles
    shown = profiles if n_rivals == 1 else [(0.1, 0.1), (0.1, 0.4), (0.4, 0.9)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2), sharex=True)
    for r, v in enumerate(VALUES):
        for c, prof in enumerate(shown):
            row = next(x for x in rows if x["value"] == v and tuple(x["rival_bids"]) == prof)
            ax = axes[r, c]
            ax.plot(BIDS, row["curve"], color="#1f5aa6", lw=1.6, label="Utility of bidder 1")
            ax.axvline(v, color="#c0392b", lw=1, ls=":", label="b = v")
            ax.axvline(max(prof), color="grey", lw=0.8, ls="--", label="Highest rival bid")
            ax.axhline(0, color="grey", lw=0.6)
            ax.plot(v, row["utility_truthful"], "o", color="#c0392b", ms=4)
            if r == 0:
                label = ", ".join(f"{p}" for p in prof)
                ax.set_title(f"Rival bid{'s' if n_rivals > 1 else ''} = {label}")
            if c == 0:
                ax.set_ylabel(f"v = {v}\nUtility of bidder 1")
            if r == 2:
                ax.set_xlabel("Bid of bidder 1, b")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.85, 0.85)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(
        f"Second-price auction against fixed rival bid{'s' if n_rivals > 1 else ''} "
        f"({n_rivals} rival{'s' if n_rivals > 1 else ''})",
        y=1.0,
    )
    fig.tight_layout()
    save_fig(fig, f"part1B_grid_{n_rivals}rival.png")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auctions", type=int, default=100_000, help="auctions simulated per bid")
    ap.add_argument(
        "--independent",
        action="store_true",
        help="draw fresh rival bids for every bid instead of common random numbers",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    out = {"auctions_per_bid": args.auctions, "common_random_numbers": not args.independent}
    for n_rivals in (1, 2):
        out[f"A_{n_rivals}rival"] = experiment_a(n_rivals, args.auctions, rng, not args.independent)
        out[f"B_{n_rivals}rival"] = experiment_b(n_rivals)

    save_json(out, "part1_results.json")

    print("Part 1A / 1C-A: best simulated bid vs. true value")
    for key in ("A_1rival", "A_2rival"):
        for r in out[key]:
            print(
                f"  rivals={r['n_rivals']} v={r['value']:.1f} {r['strategy']:>8}: "
                f"argmax b in [{r['best_bid_set_min']:.2f}, {r['best_bid_set_max']:.2f}], "
                f"u(v)={r['utility_at_truthful_bid']:.4f} (exact {r['exact_utility_at_truthful_bid']:.4f}), "
                f"max|sim-exact|={r['max_abs_error_vs_exact']:.4f}, truthful optimal={r['truthful_is_optimal']}"
            )
    print("Part 1B / 1C-B: fixed rival bids")
    for key in ("B_1rival", "B_2rival"):
        for r in out[key]:
            print(
                f"  v={r['value']:.1f} rivals={r['rival_bids']}: u(v)={r['utility_truthful']:+.2f}, "
                f"max={r['max_utility']:+.2f}, optimal b in [{r['optimal_bid_range'][0]:.2f}, "
                f"{r['optimal_bid_range'][1]:.2f}], truthful optimal={r['truthful_is_optimal']}"
            )


if __name__ == "__main__":
    main()
