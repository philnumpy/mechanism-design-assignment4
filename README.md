# Assignment 4: Auctions and the VCG Mechanism

**Team:** Vaibhav Garg, Aditya Bansal, Meet Tilala

This repository contains our code for Assignment 4 and the report built from its output ([report/report.pdf](report/report.pdf)).

| Part | Script | What it does |
| --- | --- | --- |
| 1 | [src/part1_second_price.py](src/part1_second_price.py) | Second-price sealed-bid auction. Part A sweeps bidder 1's bid against random rivals, Part B fixes the rival bids, and Part C repeats both with two rivals. |
| 2 | [src/part2_first_price.py](src/part2_first_price.py) | First-price auction with n = 2, 3, 4 bidders. Bidder 1 bids k·v while the others play ((n−1)/n)·v. Reports bidder 1's expected utility and how often the winner could have paid less and still won. |
| 3 | [src/part3_vcg_path.py](src/part3_vcg_path.py) | VCG (Clarke pivot) mechanism that buys the cheapest A→F path in the network of Figure 7.3. Reproduces the payments and sweeps misreported costs from 0 to 10. |

## How to run

```bash
pip install -r requirements.txt
python run_all.py            # runs all three parts
# or run one part on its own:
cd src
python part1_second_price.py [--auctions 100000] [--independent]
python part2_first_price.py  [--auctions 200000]
python part3_vcg_path.py
```

All random numbers come from `numpy.random.default_rng` with a fixed seed (see [src/common.py](src/common.py)), so every run reproduces the same results.

## Output

- `results/figures/`: every plot used in the report. This includes the nine individual Part 1 plots for each rival count (`part1A_<r>rival_v<v>_<strategy>.png`) and the 3×3 summary grids.
- `results/data/`: every number in the report, saved as JSON (`part1_results.json`, `part2_results.json`, `part3_results.json`).

## Layout

```
mechanism_design/
├── run_all.py
├── requirements.txt
├── src/
│   ├── common.py              # paths, seed, plotting style
│   ├── part1_second_price.py
│   ├── part2_first_price.py
│   └── part3_vcg_path.py
├── results/
│   ├── figures/
│   └── data/
└── report/
    ├── report.tex
    └── report.pdf
```
