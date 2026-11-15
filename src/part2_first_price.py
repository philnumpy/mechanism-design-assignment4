"""Part 2: first-price sealed-bid auction with n bidders.

All values are i.i.d. U[0, 1]. Bidders 2..n play the symmetric Bayes-Nash
equilibrium bid ((n - 1) / n) * v. Bidder 1 deviates and bids k * v for k on a
grid from 0.1 to 2.0. The highest bid wins and pays its own bid.

For each n in {2, 3, 4} and each k we estimate
  * bidder 1's expected utility E[(v1 - k v1) * 1{bidder 1 wins}], and
  * how often the winner "could have paid less and still won", i.e. the winning
    bid is strictly above the second-highest bid. We also report the share of
    auctions where the gap exceeds one bid increment (0.01) and the average gap
    (money left on the table).
"""

import argparse

import numpy as np

from common import SEED, save_fig, save_json, plt

N_VALUES = [2, 3, 4]
# 0.10, 0.15, ..., 2.00 plus k = 2/3 so that every equilibrium (n-1)/n is on the grid
K_GRID = np.unique(np.concatenate([np.round(np.arange(0.1, 2.0001, 0.05), 2), [2 / 3]]))
BID_INCREMENT = 0.01
COLORS = {2: "#1f5aa6", 3: "#c0392b", 4: "#2e8b57"}


def analytic_utility(k, n, grid=20001):
    """E[u1(k)] = int_0^1 (1 - k) v * min(1, k v n / (n - 1))^(n - 1) dv."""
    v = np.linspace(0.0, 1.0, grid)
    p_win = np.minimum(1.0, k * v * n / (n - 1)) ** (n - 1)
    return float(np.trapezoid((1 - k) * v * p_win, v))


def simulate(n, n_auctions, rng):
    """Common random numbers: the same value draws are reused for every k."""
    values = rng.uniform(0.0, 1.0, size=(n_auctions, n))
    v1 = values[:, 0]
    others = (n - 1) / n * values[:, 1:]
    top_other = others.max(axis=1)
    # second-highest among the other bidders (used when an "other" bidder wins)
    second_other = np.sort(others, axis=1)[:, -2] if n > 2 else np.zeros(n_auctions)

    rows = []
    for k in K_GRID:
        b1 = k * v1
        b1_wins = b1 > top_other
        u = np.where(b1_wins, v1 - b1, 0.0)

        win_bid = np.where(b1_wins, b1, top_other)
        second_bid = np.where(b1_wins, top_other, np.maximum(b1, second_other))
        gap = win_bid - second_bid  # how much less the winner could have paid

        gap_b1 = gap[b1_wins]
        rows.append(
            {
                "k": float(k),
                "expected_utility": float(u.mean()),
                "se": float(u.std(ddof=1) / np.sqrt(n_auctions)),
                "expected_utility_exact": analytic_utility(k, n),
                "p_bidder1_wins": float(b1_wins.mean()),
                "frac_winner_could_pay_less": float((gap > 0).mean()),
                "frac_gap_above_increment": float((gap > BID_INCREMENT).mean()),
                "mean_gap": float(gap.mean()),
                "frac_bidder1_could_pay_less_given_win": float((gap_b1 > 0).mean()) if gap_b1.size else 0.0,
                "mean_gap_bidder1_given_win": float(gap_b1.mean()) if gap_b1.size else 0.0,
            }
        )
    return rows


def plot(results):
    k_fine = np.linspace(0.1, 2.0, 400)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for n in N_VALUES:
        rows = results[n]
        ks = np.array([r["k"] for r in rows])
        mu = np.array([r["expected_utility"] for r in rows])
        se = np.array([r["se"] for r in rows])
        ax.errorbar(ks, mu, yerr=1.96 * se, fmt="o", ms=3, color=COLORS[n], label=f"n = {n} (simulated)")
        ax.plot(k_fine, [analytic_utility(k, n, 4001) for k in k_fine], color=COLORS[n], lw=1, alpha=0.8)
        ax.axvline((n - 1) / n, color=COLORS[n], lw=0.8, ls=":")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Bid multiplier k (bidder 1 bids k·v)")
    ax.set_ylabel("Expected utility of bidder 1")
    ax.set_title("First-price auction: bidder 1 deviates, others play ((n−1)/n)·v\n"
                 "points = simulation (95% CI), lines = closed form, dotted = equilibrium k")
    ax.set_ylim(-0.52, 0.2)
    ax.legend()
    save_fig(fig, "part2_expected_utility.png")

    # zoom near the optimum
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for n in N_VALUES:
        rows = [r for r in results[n] if 0.3 <= r["k"] <= 1.05]
        ks = np.array([r["k"] for r in rows])
        mu = np.array([r["expected_utility"] for r in rows])
        se = np.array([r["se"] for r in rows])
        ax.errorbar(ks, mu, yerr=1.96 * se, fmt="o", ms=3, color=COLORS[n], label=f"n = {n}")
        kk = np.linspace(0.3, 1.05, 200)
        ax.plot(kk, [analytic_utility(k, n, 4001) for k in kk], color=COLORS[n], lw=1, alpha=0.8)
        ax.axvline((n - 1) / n, color=COLORS[n], lw=0.8, ls=":")
    ax.set_xlabel("Bid multiplier k")
    ax.set_ylabel("Expected utility of bidder 1")
    ax.set_title("Zoom on 0.3 ≤ k ≤ 1.05: the peak sits at k = (n−1)/n")
    ax.legend()
    save_fig(fig, "part2_expected_utility_zoom.png")

    # money left on the table
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for n in N_VALUES:
        ks = [r["k"] for r in results[n]]
        axes[0].plot(ks, [r["frac_gap_above_increment"] for r in results[n]], "o-", ms=3, color=COLORS[n], label=f"n = {n}")
        axes[1].plot(ks, [r["mean_gap"] for r in results[n]], "o-", ms=3, color=COLORS[n], label=f"n = {n}")
    for ax in axes:
        ax.set_xlabel("Bid multiplier k of bidder 1")
        ax.legend()
    axes[0].set_ylabel("Share of auctions")
    axes[0].set_title("Winner could have bid more than 0.01 less and still won")
    axes[1].set_ylabel("Winning bid − second-highest bid")
    axes[1].set_title("Average amount the winner could have saved")
    fig.tight_layout()
    save_fig(fig, "part2_overpayment.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auctions", type=int, default=200_000, help="auctions simulated per (n, k)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED + 2)

    results = {n: simulate(n, args.auctions, rng) for n in N_VALUES}
    plot(results)

    summary = {}
    for n in N_VALUES:
        rows = results[n]
        best = max(rows, key=lambda r: r["expected_utility"])
        k_eq = (n - 1) / n
        eq = min(rows, key=lambda r: abs(r["k"] - k_eq))
        k_star_exact = max(np.linspace(0.1, 2.0, 1901), key=lambda k: analytic_utility(k, n, 4001))
        summary[n] = {
            "k_equilibrium": k_eq,
            "best_k_on_grid_simulated": best["k"],
            "best_utility_simulated": best["expected_utility"],
            "k_star_closed_form": float(k_star_exact),
            "utility_at_equilibrium_closed_form": 1 / (n * (n + 1)),
            "grid_point_nearest_equilibrium": eq,
        }
        print(f"n={n}: best k on grid = {best['k']:.2f} (u={best['expected_utility']:.4f}); "
              f"closed-form optimum k*={k_star_exact:.3f}, equilibrium k={k_eq:.3f}, "
              f"u_eq=1/(n(n+1))={1 / (n * (n + 1)):.4f}")
        for r in rows:
            if abs(r["k"] * 10 - round(r["k"] * 10)) < 1e-9 or abs(r["k"] - k_eq) < 0.026:
                print(f"   k={r['k']:.2f} u={r['expected_utility']:+.4f}±{1.96 * r['se']:.4f} "
                      f"(exact {r['expected_utility_exact']:+.4f})  P(win)={r['p_bidder1_wins']:.3f}  "
                      f"could-pay-less={r['frac_winner_could_pay_less']:.4f}  "
                      f"gap>0.01={r['frac_gap_above_increment']:.4f}  mean gap={r['mean_gap']:.4f}")

    save_json({"auctions_per_k": args.auctions, "k_grid": K_GRID.tolist(),
               "results": {str(n): results[n] for n in N_VALUES},
               "summary": {str(n): summary[n] for n in N_VALUES}}, "part2_results.json")


if __name__ == "__main__":
    main()
