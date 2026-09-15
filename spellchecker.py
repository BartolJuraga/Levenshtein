"""
spellchecker.py
================
Demonstracija primjene Levenshteinove i Damerau-Levenshteinove udaljenosti
na stvarnom problemu: ispravljanje pogrešaka nastalih pri tipkanju
(autokorekcija / spell-checking), po uzoru na pristup P. Norviga (2007)
"How to Write a Spelling Corrector", prošireno usporedbom nekoliko metrika
udaljenosti/sličnosti (Levenshtein, Damerau-Levenshtein, Hamming,
Jaro-Winkler).

Sustav radi u dva koraka:
    1. Iz rječnika (popis ispravnih riječi) generiraju se kandidati koji su
       na udaljenosti 0, 1 ili 2 uređivačke operacije od upisane riječi.
    2. Među kandidatima koji postoje u rječniku bira se onaj s najmanjom
       udaljenošću; ako ih je više s istom udaljenošću, bira se
       najfrekventniji (na temelju jednostavnog jezičnog modela).

Rječnik korišten u evaluaciji (`wordlist.txt`) sadrži 3000 riječi nasumično
odabranih (fiksni random seed, radi ponovljivosti) iz standardnog Hunspell
en_US rječnika (paket `hunspell-en-us`, isti rječnik koji se koristi u
stvarnim sustavima poput LibreOfficea i preglednika temeljenih na
Chromiumu/Firefoxu) - nakon filtriranja na isključivo mala slova bez
posebnih znakova, od ukupno dostupnih 48 435 takvih riječi.

Testni skup pogrešno napisanih riječi generira se **programski**, a ne
ručno: za 200 nasumično odabranih riječi iz rječnika (duljine ≥ 4 znaka)
primjenjuje se točno jedna od četiri vrste pogreške koje je opisao F. J.
Damerau (1964.) - izostavljanje, umetanje, zamjena ili transpozicija znaka
- ravnomjerno raspoređenih (po 50 primjera svake vrste), uz fiksni random
seed radi ponovljivosti rezultata.

Skripta se može pokrenuti izravno; tada se pokreće evaluacija nad ovako
generiranim skupom testnih parova (pogrešno napisana riječ, ispravna
riječ) i ispisuje se točnost (accuracy) za svaku od uspoređivanih metoda.
"""

from __future__ import annotations

import os
import random
import string
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from levenshtein import levenshtein_distance, damerau_levenshtein_distance
from metrics import hamming_distance, jaro_winkler_similarity

ALPHABET = string.ascii_lowercase
HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST_PATH = os.path.join(HERE, "wordlist.txt")


def edits1(word: str) -> Set[str]:
    """Generira sve nizove udaljene jednu uređivačku operaciju (umetanje,
    brisanje, zamjena ili transpozicija susjednih znakova) od zadane riječi.
    Broj kandidata je reda veličine O(54 * n) za riječ duljine n."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in ALPHABET]
    inserts = [l + c + r for l, r in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str) -> Set[str]:
    """Svi nizovi udaljeni dvije uređivačke operacije (primjena edits1
    dva puta)."""
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


class SpellChecker:
    """Jednostavan spell-checker temeljen na rječniku i jezičnom modelu
    učestalosti riječi."""

    def __init__(self, word_frequencies: Dict[str, int]):
        self.freq = word_frequencies
        self.vocabulary: Set[str] = set(word_frequencies.keys())
        self.total = sum(word_frequencies.values())

    @classmethod
    def from_corpus(cls, words: Iterable[str]) -> "SpellChecker":
        return cls(dict(Counter(w.lower() for w in words)))

    def probability(self, word: str) -> float:
        return self.freq.get(word, 0) / self.total if self.total else 0.0

    def known(self, words: Iterable[str]) -> Set[str]:
        return {w for w in words if w in self.vocabulary}

    def candidates(self, word: str) -> Set[str]:
        """Kandidati za ispravak, prioritet: točna riječ > udaljenost 1 >
        udaljenost 2 > nepromijenjena riječ (ako ništa bolje nije nađeno)."""
        return (
            self.known([word])
            or self.known(edits1(word))
            or self.known(edits2(word))
            or {word}
        )

    def correct(self, word: str) -> str:
        """Vraća najvjerojatniju ispravku riječi (Norvigov pristup: najprije
        minimalna udaljenost uređivanja, zatim najveća učestalost). Kandidati
        se prije odabira sortiraju abecedno kako bi odabir kod izjednačenja
        učestalosti bio deterministički i ponovljiv (iteracija preko `set`-a
        u Pythonu inače ovisi o nasumičnom hash-seedu procesa)."""
        word = word.lower()
        return max(sorted(self.candidates(word)), key=self.probability)

    # --- alternativne metode ispravka, za usporedbu metrika u radu -------

    def correct_by_metric(self, word: str, metric: str = "levenshtein") -> Optional[str]:
        """Ispravlja riječ tako da nad CIJELIM rječnikom traži riječ s
        minimalnom udaljenošću (odn. maksimalnom sličnošću) prema zadanoj
        metrici. Sporije od candidates()/correct(), ali omogućuje izravnu
        usporedbu kvalitete različitih metrika opisanih u radu.
        Zbog vremena izvođenja, kod Hammingove udaljenosti razmatraju se
        samo riječi jednake duljine. Rječnik se prolazi u abecednom
        (sortiranom) redoslijedu kako bi odabir kod izjednačenja rezultata
        bio deterministički i ponovljiv."""
        word = word.lower()
        if word in self.vocabulary:
            return word

        if metric == "hamming":
            best_score, best_word = None, None
            for w in sorted(self.vocabulary):
                if len(w) != len(word):
                    continue
                score = hamming_distance(word, w)
                if best_score is None or score < best_score:
                    best_score, best_word = score, w
            return best_word

        best_score, best_word = None, None
        for w in sorted(self.vocabulary):
            if metric == "levenshtein":
                score = levenshtein_distance(word, w)
                better = best_score is None or score < best_score
            elif metric == "damerau":
                score = damerau_levenshtein_distance(word, w)
                better = best_score is None or score < best_score
            elif metric == "jaro_winkler":
                score = jaro_winkler_similarity(word, w)
                better = best_score is None or score > best_score
            else:
                raise ValueError(f"Nepoznata metrika: {metric}")
            if better:
                best_score, best_word = score, w
        return best_word


# --------------------------------------------------------------------------
# Rječnik: 3000 riječi iz stvarnog Hunspell (en_US) rječnika
# --------------------------------------------------------------------------

def load_vocabulary(path: str = WORDLIST_PATH) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------------------------------
# Programsko generiranje testnog skupa (pogrešno napisana riječ -> ispravna)
# --------------------------------------------------------------------------

def _apply_deletion(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1:]


def _apply_insertion(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word) + 1)
    c = rng.choice(ALPHABET)
    return word[:i] + c + word[i:]


def _apply_substitution(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    c = rng.choice([x for x in ALPHABET if x != word[i]])
    return word[:i] + c + word[i + 1:]


def _apply_transposition(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


_ERROR_TYPES = {
    "brisanje": _apply_deletion,
    "umetanje": _apply_insertion,
    "zamjena": _apply_substitution,
    "transpozicija": _apply_transposition,
}


def generate_test_pairs(
    vocabulary: List[str],
    n_per_type: int = 50,
    min_len: int = 4,
    seed: int = 7,
) -> List[Tuple[str, str, str]]:
    """Programski generira testne parove (pogrešno napisana riječ, ispravna
    riječ, vrsta pogreške) primjenom točno jedne od četiri vrste pogreške
    (Damerau, 1964.) na nasumično odabrane riječi iz rječnika. Riječi kod
    kojih bi pogrešno napisan oblik slučajno opet bio ispravna riječ iz
    rječnika preskaču se, radi jednoznačnosti evaluacije."""
    rng = random.Random(seed)
    candidates = [w for w in vocabulary if len(w) >= min_len]
    rng.shuffle(candidates)

    vocab_set = set(vocabulary)
    pairs: List[Tuple[str, str, str]] = []
    counts = {t: 0 for t in _ERROR_TYPES}
    idx = 0

    for error_type, fn in _ERROR_TYPES.items():
        while counts[error_type] < n_per_type and idx < len(candidates):
            word = candidates[idx]
            idx += 1
            misspelled = fn(word, rng)
            if misspelled == word or misspelled in vocab_set:
                continue
            pairs.append((misspelled, word, error_type))
            counts[error_type] += 1

    rng.shuffle(pairs)
    return pairs


def evaluate_norvig_style(sc: SpellChecker, pairs) -> float:
    correct = 0
    for misspelled, target, *_ in pairs:
        if sc.correct(misspelled) == target:
            correct += 1
    return correct / len(pairs)


def evaluate_by_metric(sc: SpellChecker, pairs, metric: str) -> Tuple[float, float]:
    correct = 0
    start = time.perf_counter()
    for misspelled, target, *_ in pairs:
        guess = sc.correct_by_metric(misspelled, metric=metric)
        if guess == target:
            correct += 1
    elapsed = time.perf_counter() - start
    return correct / len(pairs), elapsed


if __name__ == "__main__":
    vocabulary = load_vocabulary()
    sc = SpellChecker.from_corpus(vocabulary)
    test_pairs = generate_test_pairs(vocabulary, n_per_type=50, seed=7)

    print(f"Rječnik: {len(vocabulary)} riječi (uzorak iz Hunspell en_US rječnika)")
    print(f"Testni skup: {len(test_pairs)} programski generiranih primjera "
          f"({len(_ERROR_TYPES)} vrste pogreške x 50)\n")

    print("=== Primjeri ispravaka (Norvigov pristup, edits1/edits2) ===")
    for misspelled, target, err_type in test_pairs[:10]:
        guess = sc.correct(misspelled)
        mark = "OK" if guess == target else "POGREŠNO"
        print(f"  [{err_type:13s}] {misspelled!r:15} -> {guess!r:15} "
              f"(očekivano: {target!r:15}) [{mark}]")

    acc_norvig = evaluate_norvig_style(sc, test_pairs)
    print(f"\nTočnost (Norvigov pristup, edits1/edits2 + učestalost): {acc_norvig:.2%}")

    print("\n=== Usporedba metrika na cijelom rječniku (brute-force potraga) ===")
    for metric in ["levenshtein", "damerau", "hamming", "jaro_winkler"]:
        acc, elapsed = evaluate_by_metric(sc, test_pairs, metric)
        print(f"  {metric:14s}: točnost={acc:.2%}   ukupno vrijeme={elapsed:.4f}s "
              f"({elapsed/len(test_pairs)*1000:.3f} ms/riječ)")

    print("\n=== Točnost po vrsti pogreške (Norvigov pristup) ===")
    for err_type in _ERROR_TYPES:
        subset = [p for p in test_pairs if p[2] == err_type]
        acc = evaluate_norvig_style(sc, subset)
        print(f"  {err_type:14s}: {acc:.2%}  (n={len(subset)})")
