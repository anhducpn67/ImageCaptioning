from bean.Container import Container
from pipeline.prepare_dataset import PrepareDataset


def run():
    container = Container()
    container = PrepareDataset(container).execute()


if __name__ == '__main__':
    run()
