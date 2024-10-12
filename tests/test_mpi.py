import unittest
import numpy as np

from gx_features.mpi import (
    proc0_print,
    distribute_work_mpi,
    join_dataframes_mpi,
)


class Tests(unittest.TestCase):
    def test_proc0_print(self):
        proc0_print("Hello, world!")
        proc0_print("Hello, world!", 2, 3.3)
        