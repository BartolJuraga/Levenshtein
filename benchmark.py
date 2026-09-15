"""
benchmark.py
============
Empirijska analiza vremenske složenosti implementiranih algoritama.
Mjeri se vrijeme izvođenja levenshtein_distance, levenshtein_distance_rows
i damerau_levenshtein_distance za rastuće duljine slučajno generiranih
nizova, radi potvrde teorijske O(n*m) složenosti.

Rezultati se ispisuju u konzolu i spremaju u benchmark_results.csv te
grafički prikazuju u benchmark_plot.png (koristi se u radu kao Slika).
"""

import csv
import random
import string
import time

from levenshtein import (
    levenshtein_distance,
    levenshtein_distance_rows,
    damerau_levenshtein_distance,
)


def random_string(n: int, alphabet: str = string.ascii_lowercase) -> str:
    return "".join(random.choice(alphabet) for _ in range(n))


def time_function(func, a: str, b: str, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(a, b)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


def run_benchmark(sizes):
    random.seed(42)
    rows = []
    for n in sizes:
        a = random_string(n)
        b = random_string(n)
        t_full = time_function(levenshtein_distance, a, b)
        t_rows = time_function(levenshtein_distance_rows, a, b)
        t_dam = time_function(damerau_levenshtein_distance, a, b)
        rows.append(
            {
                "n": n,
                "levenshtein_full_s": t_full,
                "levenshtein_rows_s": t_rows,
                "damerau_levenshtein_s": t_dam,
            }
        )
        print(
            f"n={n:5d}  lev_full={t_full:9.6f}s  lev_rows={t_rows:9.6f}s  "
            f"damerau={t_dam:9.6f}s"
        )
    return rows


def save_csv(rows, path="benchmark_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path="benchmark_plot.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nije dostupan - preskačem izradu grafa.")
        return

    ns = [r["n"] for r in rows]
    lev = [r["levenshtein_full_s"] for r in rows]
    lev_rows = [r["levenshtein_rows_s"] for r in rows]
    dam = [r["damerau_levenshtein_s"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, lev, marker="o", label="Levenshtein (puna matrica)")
    plt.plot(ns, lev_rows, marker="s", label="Levenshtein (dva retka)")
    plt.plot(ns, dam, marker="^", label="Damerau-Levenshtein")
    plt.xlabel("Duljina nizova n (n = m)")
    plt.ylabel("Vrijeme izvođenja (s)")
    plt.title("Empirijska vremenska složenost implementiranih algoritama")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Graf spremljen u {path}")


if __name__ == "__main__":
    sizes = [50, 100, 200, 400, 800, 1200]
    rows = run_benchmark(sizes)
    save_csv(rows)
    save_plot(rows)
