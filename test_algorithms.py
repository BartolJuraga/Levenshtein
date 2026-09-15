"""
test_algorithms.py
===================
Jedinični testovi za provjeru ispravnosti implementacija u levenshtein.py
i metrics.py. Pokreće se s: python3 -m pytest test_algorithms.py -v
(ili izravno: python3 test_algorithms.py)
"""

import unittest

from levenshtein import (
    levenshtein_distance,
    levenshtein_distance_rows,
    levenshtein_distance_ops,
    damerau_levenshtein_distance,
    optimal_string_alignment,
)
from metrics import (
    hamming_distance,
    lcs_length,
    lcs_distance,
    jaro_similarity,
    jaro_winkler_similarity,
)


class TestLevenshtein(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(levenshtein_distance("test", "test"), 0)

    def test_empty_strings(self):
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_classic_example_kitten_sitting(self):
        # poznati udžbenički primjer: udaljenost je 3
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_single_substitution(self):
        self.assertEqual(levenshtein_distance("cat", "bat"), 1)

    def test_single_insertion(self):
        self.assertEqual(levenshtein_distance("cat", "cats"), 1)

    def test_symmetry(self):
        a, b = "informatika", "infromatika"
        self.assertEqual(
            levenshtein_distance(a, b), levenshtein_distance(b, a)
        )

    def test_triangle_inequality(self):
        a, b, c = "kloboučnik", "kloboucnik", "kobouk"
        self.assertLessEqual(
            levenshtein_distance(a, c),
            levenshtein_distance(a, b) + levenshtein_distance(b, c),
        )

    def test_rows_matches_full_matrix(self):
        pairs = [("kitten", "sitting"), ("", "abc"), ("varazdin", "varaždin"), ("foi", "foi")]
        for a, b in pairs:
            self.assertEqual(
                levenshtein_distance(a, b), levenshtein_distance_rows(a, b)
            )

    def test_ops_reconstruction_matches_distance(self):
        a, b = "kitten", "sitting"
        dist, ops = levenshtein_distance_ops(a, b)
        self.assertEqual(dist, levenshtein_distance(a, b))
        self.assertEqual(len(ops), dist)


class TestDamerauLevenshtein(unittest.TestCase):
    def test_transposition_counts_as_one(self):
        # "ab" -> "ba" je jedna transpozicija: Damerau-Levenshtein = 1,
        # dok je klasični Levenshtein = 2 (dvije zamjene)
        self.assertEqual(damerau_levenshtein_distance("ab", "ba"), 1)
        self.assertEqual(levenshtein_distance("ab", "ba"), 2)

    def test_identical_strings(self):
        self.assertEqual(damerau_levenshtein_distance("foi", "foi"), 0)

    def test_never_greater_than_levenshtein(self):
        pairs = [("kitten", "sitting"), ("informatika", "infromatika"), ("abcdef", "badcfe")]
        for a, b in pairs:
            self.assertLessEqual(
                damerau_levenshtein_distance(a, b), levenshtein_distance(a, b)
            )

    def test_true_damerau_vs_osa_alignment_example(self):
        # klasični primjer razlike prave Damerau-Levenshtein udaljenosti
        # i OSA udaljenosti: "CA" -> "ABC"
        self.assertEqual(damerau_levenshtein_distance("CA", "ABC"), 2)
        self.assertEqual(optimal_string_alignment("CA", "ABC"), 3)


class TestAlternativeMetrics(unittest.TestCase):
    def test_hamming_equal_length(self):
        self.assertEqual(hamming_distance("karolin", "kathrin"), 3)

    def test_hamming_requires_equal_length(self):
        with self.assertRaises(ValueError):
            hamming_distance("abc", "ab")

    def test_lcs_length_basic(self):
        self.assertEqual(lcs_length("ABCBDAB", "BDCABA"), 4)

    def test_lcs_distance_relation(self):
        a, b = "ABCBDAB", "BDCABA"
        expected = len(a) + len(b) - 2 * lcs_length(a, b)
        self.assertEqual(lcs_distance(a, b), expected)

    def test_jaro_identical(self):
        self.assertEqual(jaro_similarity("foi", "foi"), 1.0)

    def test_jaro_winkler_known_value(self):
        # poznati primjer iz literature: MARTHA vs MARHTA
        val = jaro_winkler_similarity("MARTHA", "MARHTA")
        self.assertAlmostEqual(val, 0.961, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
