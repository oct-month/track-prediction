from data_loader import data_iter_pre
from config import batch_size


def main():
    data_iter_pre(batch_size)
    print('\ndata iter pre done.')


if __name__ == '__main__':
    main()
