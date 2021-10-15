from data_loader import data_track_iter

if __name__ == '__main__':
    max_n, min_n = 0, 10000000
    for track in data_track_iter():
        n = 0  
        f = None
        for t in track:
            x = t.to_tuple()
            if f is None:
                f = x
            else:
                if f == x:
                    n += 1
                else:
                    max_n = max(max_n, n)
                    min_n = min(min_n, n)
                    n = 0
    print(max_n, min_n)
