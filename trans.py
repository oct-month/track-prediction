if __name__ == '__main__':
    with open('data/20210401.txt', 'r', encoding='UTF-8') as f:
        s = f.read()
        s = s.replace('\t', '","')
        s = s.replace('\n', '"\r\n"')
        if not s.startswith('"'):
            s = '"' + s
        if not s.endswith('"'):
            s = s + '"'

    with open('data/20210401.csv', 'w', encoding='GBK') as f:
        f.write(s)
