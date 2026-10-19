import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from hadoop_worker import (clean_map, clean_reduce, parse_record, repair_record,
                           score_map, score_reduce)


def run_worker(function, input_text, *args):
    output = io.StringIO()
    with patch.object(sys, "stdin", io.StringIO(input_text)), redirect_stdout(output):
        function(*args)
    return output.getvalue()


class QualityRulesTest(unittest.TestCase):
    def test_postal_code_keeps_leading_zero_and_separate_rating_events(self):
        row, key, error = parse_record("users", "1::F::25::12::02138")
        self.assertIsNone(error)
        self.assertEqual(row, "1::F::25::12::02138")
        first = parse_record("ratings", "1::2::4::978300000")
        second = parse_record("ratings", "1::2::5::978300001")
        self.assertNotEqual(first[1], second[1])

    def test_rejects_invalid_domain_without_guessing_repair(self):
        self.assertEqual(parse_record("ratings", "1::2::6::978300000")[2], "rating_domain")
        self.assertEqual(parse_record("movies", "2::Film (2000)::Unknown")[2], "genre_domain")

    def test_repairs_only_observed_unambiguous_corruption(self):
        cases = [
            ("users", "3036:M:56:13:14843", "3036::M::56::13::14843", "single_colon_delimiter"),
            ("movies", "32|Twelve Monkeys (1995)|Drama|Sci-Fi",
             "32::Twelve Monkeys (1995)::Drama|Sci-Fi", "pipe_delimiter"),
            ("ratings", "1,2,5,978300000", "1::2::5::978300000", "comma_delimiter"),
            ("ratings", "1::2::five::978300000", "1::2::5::978300000", "english_five"),
            ("ratings", "1::2::5::978300000000", "1::2::5::978300000", "milliseconds_to_seconds"),
            ("ratings", "1::2::5::978300000::EXTRA_FIELD", "1::2::5::978300000", "drop_literal_extra_field"),
        ]
        for table, raw, expected, reason in cases:
            with self.subTest(table=table, reason=reason):
                row, _key, error, repairs = repair_record(table, raw)
                self.assertIsNone(error)
                self.assertEqual(row, expected)
                self.assertIn(reason, repairs)

    def test_ambiguous_values_remain_quarantined(self):
        for raw in ("1::2::3.5::978300000", "1::2::::978300000",
                    "1::2::6::978300000", "1::2::5::99"):
            with self.subTest(raw=raw):
                self.assertIsNotNone(repair_record("ratings", raw)[2])

    def test_conflicting_logical_key_is_quarantined(self):
        mapped = run_worker(clean_map,
                            "1::2::4::978300000\n1::2::5::978300000\n",
                            "ratings", {"1"}, {"2"})
        reduced = run_worker(clean_reduce, mapped)
        actions = [json.loads(line) for line in reduced.splitlines()]
        self.assertEqual([x["action"] for x in actions], ["quarantine", "quarantine"])
        self.assertTrue(all(x["reason"] == "key_conflict" for x in actions))

    def test_deduplication_prefers_canonical_source_independent_of_input_order(self):
        rows = "1::2::five::978300000\n1::2::5::978300000\n"
        for input_rows in (rows, "1::2::5::978300000\n1::2::five::978300000\n"):
            mapped = run_worker(clean_map, input_rows, "ratings", {"1"}, {"2"})
            reduced = run_worker(clean_reduce, "\n".join(sorted(mapped.splitlines())) + "\n")
            items = [json.loads(line) for line in reduced.splitlines()]
            kept = [item for item in items if item["action"] == "keep"]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["raw"], "1::2::5::978300000")
            self.assertEqual(len([item for item in items if item["action"] == "deduplicate"]), 1)

    def test_score_reducer_counts_exact_duplicates(self):
        mapped = run_worker(score_map,
                            "1::2::4::978300000\n1::2::4::978300000\n",
                            "ratings", {"1"}, {"2"}, 978300100)
        reduced = run_worker(score_reduce, "\n".join(sorted(mapped.splitlines())) + "\n")
        counts = {}
        for line in reduced.splitlines():
            key, value = line.split("\t")
            counts[key] = counts.get(key, 0) + int(value)
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["duplicate_excess"], 1)
        self.assertEqual(counts["consistent"], 2)


if __name__ == "__main__":
    unittest.main()
