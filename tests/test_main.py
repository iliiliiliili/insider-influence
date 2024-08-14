import json
import unittest
import torch
import pandas as pd
from pathlib import Path
import os

from main import (
    main,
)


class TestMain(unittest.TestCase):

    def test_main_test(self):
        expected_results_folder = "expected_results"

        mode = "test"
        name = "original"
        path = "."
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results_folder = "results"
        models_folder = "models"
        return_results = True
        seed = 2605

        main(
            mode,
            name,
            path,
            device,
            results_folder,
            models_folder,
            return_results,
            seed,
        )

        def find_result_files(path):

            subdirs = os.walk(path)
            all_result_files = []

            for subdir, _, files in subdirs:
                for file in files:
                    if "result.json" in file:
                        all_result_files.append((subdir, file))

            print(f"Found {len(all_result_files)} results")

            return all_result_files

        expected_files = find_result_files(expected_results_folder)

        self.assertTrue(len(expected_files) > 0)

        for subdir, file in expected_files:
            expected_file_name = os.path.join(subdir, file)

            tested_filename = expected_file_name.replace(
                expected_results_folder, f"{results_folder}/{name}"
            )

            with open(expected_file_name, "r") as f:
                expected_data = json.load(f)

            with open(tested_filename, "r") as f:
                tested_data = json.load(f)

            if not isinstance(expected_data, list):
                expected_data = [expected_data]

            if not isinstance(tested_data, list):
                tested_data = [tested_data]

            self.assertEqual(len(expected_data), len(tested_data))

            for e, t in zip(expected_data, tested_data):
                for k, v in e.items():
                    if isinstance(v, float):
                        self.assertAlmostEqual(v, t[k])
                    else:
                        self.assertEqual(v, t[k])
