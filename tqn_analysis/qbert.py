if __name__ == '__main__':
    import jaynes
    from pg_experiments import instr
    from treeqn.nstep_run import train, launch
    from treeqn.config import Config

    jaynes.config("gpu-high", runner=dict(n_cpu=15))

    thunk = instr(launch)
    jaynes.run(thunk)
    jaynes.listen()
