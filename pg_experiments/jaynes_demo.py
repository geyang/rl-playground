def train():
    print('running')


if __name__ == '__main__':
    import jaynes
    from pg_experiments import RUN

    jaynes.config('gpu-mars')
    jaynes.run(train)
    jaynes.listen()
