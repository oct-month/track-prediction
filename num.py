import os

EXCLUDE_DIR = [
    '.git',
    '.venv',
    '.mypy_cache',
    '__pycache__',
    '.img',
    '.vscode',
    'datasets',
    'radardata'
]

EXCLUDE_FILE = [
    '.jpg',
    '.jpeg',
    '.png',
    '.pt',
    '.params',
    '.dll',
    '.xlsx',
    '.csv'
]

result = 0

def go(path: str) -> None:
    global result
    for e in EXCLUDE_DIR:
        if e in path:
            return
    print(path)
    for p in os.listdir(path):
        t = os.path.join(path, p)
        if os.path.isdir(t):
            go(t)
        elif os.path.isfile(t):
            if os.path.splitext(t)[-1] not in EXCLUDE_FILE:
                try:
                    with open(t, 'r', encoding='UTF-8') as f:
                        for l in f.readlines():
                            if l.strip() != '':
                                result += 1
                    print('Info: ', t)
                        # result += len(f.readlines())
                except Exception:
                    print('Error: ', t)


if __name__ == '__main__':
    go('./')
    print(result)
