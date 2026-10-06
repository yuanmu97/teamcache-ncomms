from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from teamcache.benchmark import arm_sequence, generate_shape, load_dataset


class BundledBenchmark(unittest.TestCase):
    def test_cli_reports_package_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "teamcache.cli", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "teamcache 0.3.0")

    def test_arm_order_can_be_reversed(self):
        self.assertEqual(arm_sequence("both", "lru-first"), ("lru", "teamcache"))
        self.assertEqual(
            arm_sequence("both", "teamcache-first"), ("teamcache", "lru")
        )
        self.assertEqual(arm_sequence("lru", "teamcache-first"), ("lru",))

    def test_shape_is_the_documented_case(self):
        dataset = load_dataset(None)
        rows = generate_shape(dataset)
        self.assertEqual(dataset["name"], "ring_20x5_p3000_r8")
        self.assertEqual(len(rows), 20 * 5 * 8)
        self.assertEqual(rows[0]["prompt_tokens"], 3000)
        self.assertEqual(rows[-1]["prompt_tokens"], 3000 + 39 * 144)
        self.assertEqual(
            [(r["member"], r["turn"]) for r in rows[:6]],
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1)],
        )

    def test_reported_result_is_self_consistent(self):
        path = Path(__file__).parents[1] / "benchmarks/ring_20x5/reported.json"
        result = json.loads(path.read_text())
        lru = result["lru"]["steady_prefix_hit_rate"]["mean"]
        teamcache = result["teamcache"]["steady_prefix_hit_rate"]["mean"]
        self.assertAlmostEqual(
            (teamcache - lru) * 100,
            result["comparison"]["steady_hit_rate_gain_pp"],
            places=6,
        )
        self.assertAlmostEqual(
            result["lru"]["steady_wall_s"]["mean"]
            / result["teamcache"]["steady_wall_s"]["mean"],
            result["comparison"]["steady_wall_speedup"],
            places=6,
        )

    def test_h200_opposite_order_result_is_self_consistent(self):
        path = Path(__file__).parents[1] / "benchmarks/ring_20x5/h200_glm53_420800.json"
        result = json.loads(path.read_text())
        speedups = []
        for trial in result["trials"]:
            speedup = trial["lru"]["wall_s"] / trial["teamcache"]["wall_s"]
            self.assertAlmostEqual(speedup, trial["wall_speedup"], places=9)
            speedups.append(speedup)
        self.assertAlmostEqual(
            sum(speedups) / len(speedups),
            result["summary"]["mean_wall_speedup"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
