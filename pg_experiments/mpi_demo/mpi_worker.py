from playground import mpi


def worker():
    id = mpi.tools.proc_id()
    num_procs = mpi.tools.num_procs()

    print(f'id = {id}, num_procs = {num_procs}')
