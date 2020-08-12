def train():
    from ml_logger import logger
    logger.print('running')


if __name__ == '__main__':
    import jaynes
    from pg_experiments import instr

    thunk = instr(train)
    jaynes.config('gpu-mars', launch=dict(timeout=None))
    jaynes.run(thunk)
