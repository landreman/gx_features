import unittest
import numpy as np
import pandas as pd
from mpi4py import MPI

from gx_features.mpi import (
    proc0_print,
    distribute_work_mpi,
    join_dataframes_mpi,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc0 = (rank == 0)

class Tests(unittest.TestCase):
    def test_proc0_print(self):
        proc0_print("Hello, world!")
        proc0_print("Hello, world!", 2, 3.3)
        
    def test_distribute_work_mpi(self):
        for d1 in [1, 2, 3, 4, 5, 7, 10, 21]:
            arr1 = comm.bcast(np.random.rand(d1))
            arr2 = comm.bcast(np.random.rand(d1, 10))
            arr3 = comm.bcast(np.random.rand(d1, 5, 17))

            arr1_split, arr2_split, arr3_split = distribute_work_mpi(arr1, arr2, arr3)

            n_data_per_process = d1 // size

            if proc0:
                print(f"test_distribute_work_mpi  d1={d1}  n_data_per_process={n_data_per_process}", flush=True)
                # First check the arrays on proc0:
                np.testing.assert_allclose(arr1_split, arr1[:n_data_per_process])
                np.testing.assert_allclose(arr2_split, arr2[:n_data_per_process, :])
                np.testing.assert_allclose(arr3_split, arr3[:n_data_per_process, :, :])

                # Next check the arrays on the other procs:
                for j in range(1, size):
                    arr1_split = comm.recv(source=j)
                    arr2_split = comm.recv(source=j)
                    arr3_split = comm.recv(source=j)
                    if j == size - 1:
                        # Last proc may have a slightly different number of rows:
                        np.testing.assert_allclose(arr1_split, arr1[j * n_data_per_process:])
                        np.testing.assert_allclose(arr2_split, arr2[j * n_data_per_process:, :])
                        np.testing.assert_allclose(arr3_split, arr3[j * n_data_per_process:, :, :])
                    else:
                        np.testing.assert_allclose(arr1_split, arr1[j * n_data_per_process:(j + 1) * n_data_per_process])
                        np.testing.assert_allclose(arr2_split, arr2[j * n_data_per_process:(j + 1) * n_data_per_process, :])
                        np.testing.assert_allclose(arr3_split, arr3[j * n_data_per_process:(j + 1) * n_data_per_process, :, :])

            else:
                comm.send(arr1_split, dest=0)
                comm.send(arr2_split, dest=0)
                comm.send(arr3_split, dest=0)

    def test_join_dataframes_mpi(self):
        columns = ["a", "b", "c"]
        n = len(columns)
        for d1 in [1, 2, 3, 4, 5, 7, 10, 21]:
            arr = comm.bcast(np.random.rand(d1, n))
            (arr_split,) = distribute_work_mpi(arr)

            df_split = pd.DataFrame(arr_split, columns=columns)
            df_mpi = join_dataframes_mpi(df_split)

            if proc0:
                print(df_mpi)
                df_no_mpi = pd.DataFrame(arr, columns=columns)

                np.testing.assert_allclose(df_mpi.to_numpy(), df_no_mpi.to_numpy())