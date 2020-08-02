from pg_experiments.mpi_demo.mpi_worker import worker

if __name__ == '__main__':
    import jaynes

    n_cpu = 4
    run_config = {
        'n_cpu': n_cpu * 2 + 1,
        'entry_script': f"mpirun -np {n_cpu} --oversubscribe "
                        f"/pkgs/anaconda3/bin/python -u -m jaynes.entry"}

    jaynes.config(runner=run_config)
    # jaynes.config('local')
    jaynes.run(worker)
    jaynes.listen()
