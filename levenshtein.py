"""
levenshtein.py
================
Implementacije Levenshteinove udaljenosti i Damerau-Levenshteinove
udaljenosti konačnih nizova znakova, izrađene za potrebe završnog rada
"Levenshteinova udaljenost konačnih nizova".

Sadrži:
    - levenshtein_distance          : klasični O(n*m) DP algoritam (puna matrica)
    - levenshtein_distance_rows     : prostorno optimizirana varijanta O(min(n,m))
    - levenshtein_distance_ops      : distanca + rekonstrukcija niza uređivačkih
                                       operacija (backtracking), korisno za prikaz
    - damerau_levenshtein_distance  : "prava" Damerau-Levenshteinova udaljenost
                                       (dopušta transpoziciju susjednih znakova,
                                       bez ograničenja da se isti podniz uređuje
                                       samo jednom)
    - optimal_string_alignment      : OSA udaljenost (pojednostavljena varijanta
                                       koja NE zadovoljava nejednakost trokuta)

Autor: izrađeno za potrebe završnog rada, FOI Varaždin, 2026.
"""

from __future__ import annotations

from typing import List, Tuple


def levenshtein_distance(a: str, b: str) -> int:
    """Klasična Levenshteinova udaljenost pomoću pune matrice dinamičkog
    programiranja. Vremenska i prostorna složenost O(n*m), gdje su n i m
    duljine ulaznih nizova a i b.

    Rekurzivna definicija (Wagner & Fischer, 1974; Levenshtein, 1965/1966):

        D[0][j] = j
        D[i][0] = i
        D[i][j] = D[i-1][j-1]                    ako je a[i-1] == b[j-1]
        D[i][j] = 1 + min(D[i-1][j],      # brisanje
                           D[i][j-1],      # umetanje
                           D[i-1][j-1])    # zamjena
                                            inače
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # brisanje znaka iz a
                d[i][j - 1] + 1,      # umetanje znaka iz b
                d[i - 1][j - 1] + cost,  # zamjena (ili podudaranje)
            )
    return d[n][m]


def levenshtein_distance_rows(a: str, b: str) -> int:
    """Prostorno optimizirana verzija Levenshteinove udaljenosti koja
    pamti samo dva retka matrice umjesto cijele matrice.
    Vremenska složenost ostaje O(n*m), a prostorna se smanjuje na
    O(min(n, m))."""
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)

    previous = list(range(m + 1))
    current = [0] * (m + 1)

    for i in range(1, n + 1):
        current[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous, current = current, previous
    return previous[m]


def levenshtein_distance_ops(a: str, b: str) -> Tuple[int, List[str]]:
    """Izračunava Levenshteinovu udaljenost i vraća listu uređivačkih
    operacija (backtracking kroz punu matricu) potrebnih za transformaciju
    niza a u niz b. Koristi se u radu za ilustraciju postupka uređivanja."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )

    # Backtracking od (n, m) prema (0, 0)
    ops: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
            continue
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(f"zamijeni '{a[i-1]}' -> '{b[j-1]}' (pozicija {i-1})")
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(f"izbriši '{a[i-1]}' (pozicija {i-1})")
            i -= 1
        else:
            ops.append(f"umetni '{b[j-1]}' (pozicija {i})")
            j -= 1

    ops.reverse()
    return d[n][m], ops


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Prava (neograničena) Damerau-Levenshteinova udaljenost: uz umetanje,
    brisanje i zamjenu dopušta i transpoziciju dvaju susjednih znakova kao
    jednu operaciju cijene 1. Za razliku od "optimal string alignment"
    varijante, ovdje se isti podniz smije uređivati više puta, čime se
    zadržava svojstvo metrike (nejednakost trokuta).

    Implementacija prati algoritam Lowrancea i Wagnera (1975), složenosti
    O(n*m) u vremenu i O(n*m) u prostoru (uz pomoćni rječnik abecede)."""
    da: dict[str, int] = {}
    n, m = len(a), len(b)
    max_dist = n + m

    d = [[0] * (m + 2) for _ in range(n + 2)]
    d[0][0] = max_dist
    for i in range(0, n + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(0, m + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    for i in range(1, n + 1):
        db = 0
        for j in range(1, m + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            if a[i - 1] == b[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1

            d[i + 1][j + 1] = min(
                d[i][j] + cost,                              # zamjena/podudaranje
                d[i + 1][j] + 1,                              # umetanje
                d[i][j + 1] + 1,                              # brisanje
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),  # transpozicija
            )
        da[a[i - 1]] = i

    return d[n + 1][m + 1]


def optimal_string_alignment(a: str, b: str) -> int:
    """OSA (optimal string alignment) udaljenost: dopušta transpoziciju
    susjednih znakova, ali svaki podniz smije biti uređen najviše jednom
    (npr. isti par znakova ne smije biti i transponiran i naknadno ponovno
    uređivan). Zbog toga OSA NE zadovoljava nejednakost trokuta i stoga
    formalno nije metrika, za razliku od prave Damerau-Levenshteinove
    udaljenosti. Vremenska i prostorna složenost O(n*m)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


if __name__ == "__main__":
    primjeri = [
        ("kitten", "sitting"),
        ("KUCA", "MACKA"),
        ("informatika", "infromatika"),  # transpozicija
    ]
    for x, y in primjeri:
        lev = levenshtein_distance(x, y)
        dam = damerau_levenshtein_distance(x, y)
        osa = optimal_string_alignment(x, y)
        print(f"{x!r} -> {y!r}: Levenshtein={lev}, Damerau-Levenshtein={dam}, OSA={osa}")
