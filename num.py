import os

EXCLUDE = [
    '.git',
    '.venv',
    '.mypy_cache',
    '__pycache__',
]

result = 0

def go(path: str) -> None:
    global result
    print(path)
    for e in EXCLUDE:
        if e in path:
            return
    for p in os.listdir(path):
        t = os.path.join(path, p)
        if os.path.isdir(t):
            go(t)
        elif os.path.isfile(t):
            if t.endswith('.py'):
                try:
                    with open(t, 'r', encoding='UTF-8') as f:
                        for l in f.readlines():
                            if l.strip() != '':
                                result += 1
                        # result += len(f.readlines())
                except Exception:
                    pass


if __name__ == '__main__':
    go('./')
    print(result)
