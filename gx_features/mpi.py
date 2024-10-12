import pandas as pd

def proc0_print(*args, **kwargs):
    try:
        from mpi4py import MPI
    except:
        MPI = None

    if MPI is None:
        print(*args, **kwargs)
    else:
        if MPI.COMM_WORLD.rank == 0:
            print(*args, **kwargs)
            

def distribute_work_mpi(*arrays):
    """
    Distribute work among MPI processes.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_data = arrays[0].shape[0]
    for arr in arrays:
        assert arr.shape[0] == n_data

    n_per_process = n_data // size

    start = rank * n_per_process
    stop = start + n_per_process
    if rank == size - 1:
        stop = n_data

    print(f"Rank {rank} will process data from {start} to {stop}")

    return (a[start:stop] for a in arrays)


def join_dataframes_mpi(df):
    """
    Join dataframes from different MPI processes.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        dataframes = [df]
        for i in range(1, size):
            data = comm.recv(source=i)
            dataframes.append(data)
        return pd.concat(dataframes, ignore_index=True)
    else:
        comm.send(df, dest=0)