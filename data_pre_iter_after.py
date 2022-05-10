import threading

from data_loader import data_iter_pre

batch_sizes = [20, 50, 100, 200, 500, 1000]

class DataThread(threading.Thread):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def run(self):
        print(f'data iter pre for batch size: {self.batch_size} start.')
        data_iter_pre(self.batch_size)
        print(f'batch size: {self.batch_size} done.')


def main():
    threads = []
    for i in batch_sizes:
        t = DataThread(i)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print('data iter pre done.')


if __name__ == '__main__':
    main()
