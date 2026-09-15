"""
metrics.py
==========
Alternativne mjere udaljenosti/sličnosti nizova, korištene u radu za
usporedbu s Levenshteinovom i Damerau-Levenshteinovom udaljenošću:

    - hamming_distance    : broj različitih znakova na istim pozicijama
                             (definirana samo za nizove jednake duljine)
    - lcs_length          : duljina najdulje zajedničke podniza (LCS)
    - lcs_distance        : udaljenost izvedena iz LCS-a (samo umetanja/brisanja)
    - jaro_similarity     : Jaro sličnost (0..1)
    - jaro_winkler_similarity : Jaro-Winkler sličnost s dodatnim bonusom za
                                zajednički prefiks (0..1)
"""

from __future__ import annotations


def hamming_distance(a: str, b: str) -> int:
    """Hammingova udaljenost: broj pozicija na kojima se znakovi razlikuju.
    Definirana je samo za nizove jednake duljine; u suprotnom se baca
    ValueError (za razliku od Levenshteinove udaljenosti, koja dopušta
    nizove različitih duljina)."""
    if len(a) != len(b):
        raise ValueError("Hammingova udaljenost zahtijeva nizove jednake duljine")
    return sum(1 for x, y in zip(a, b) if x != y)


def lcs_length(a: str, b: str) -> int:
    """Duljina najdulje zajedničke podniza (Longest Common Subsequence),
    izračunata dinamičkim programiranjem u O(n*m)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                d[i][j] = d[i - 1][j - 1] + 1
            else:
                d[i][j] = max(d[i - 1][j], d[i][j - 1])
    return d[n][m]


def lcs_distance(a: str, b: str) -> int:
    """Udaljenost izvedena iz LCS-a: minimalan broj umetanja i brisanja
    (bez zamjena) potrebnih da se niz a pretvori u niz b.
    lcs_distance(a, b) = len(a) + len(b) - 2 * lcs_length(a, b)."""
    return len(a) + len(b) - 2 * lcs_length(a, b)


def jaro_similarity(a: str, b: str) -> float:
    """Jaro sličnost dvaju nizova, vrijednost u [0, 1] (1 = identični)."""
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0

    match_distance = max(len_a, len_b) // 2 - 1
    match_distance = max(match_distance, 0)

    a_matches = [False] * len_a
    b_matches = [False] * len_b

    matches = 0
    for i in range(len_a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_b)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    m = matches
    return (m / len_a + m / len_b + (m - transpositions) / m) / 3.0


def jaro_winkler_similarity(a: str, b: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro-Winkler sličnost: Jaro sličnost uvećana bonusom za zajednički
    prefiks duljine do max_prefix znakova, s faktorom skaliranja p
    (standardno p = 0.1)."""
    jaro = jaro_similarity(a, b)
    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix += 1
            if prefix == max_prefix:
                break
        else:
            break
    return jaro + prefix * p * (1 - jaro)
