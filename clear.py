import os
import shutil


def do_clear(path: str):
    for p in os.listdir(path):
        p = os.path.join(path, p)
        if os.path.isdir(p):
            if os.path.split(p)[-1] == '__pycache__':
                shutil.rmtree(p)
            else:
                do_clear(p)


if __name__ == '__main__':
    do_clear('.')
