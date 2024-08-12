import unittest
import torch
import pandas as pd
from pathlib import Path

from main import (
    main,
)
class TestMain(unittest.TestCase):

    def test_main_test(self):
        mode = "test"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        save_folder = None

        output = main(mode, device, save_folder)

        self.assertTrue(len(output.items()) > 0)

        for name, result in output.items():
            if name in ["table_10"]:
                expected_result = pd.read_csv(Path("expected_results") / (name + ".csv"), header=[0], index_col=[0, 1], skipinitialspace=True)
                expected_result.columns = pd.to_numeric(expected_result.columns)
            else:
                expected_result = pd.read_csv(Path("expected_results") / (name + ".csv"), header=[0, 1, 2], index_col=[0, 1], skipinitialspace=True)

            print(expected_result)
            print(result)
            print(expected_result.equals(result))

            self.assertTrue(expected_result.equals(result))
