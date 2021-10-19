from data_loader import data_track_iter

if __name__ == '__main__':
    MAX_LONGI, MAX_LATI, MAX_H = .0, .0, .0
    MIN_LOGIN, MIN_LATI, MIN_H = 1e10, 1e10, 1e10
    for track in data_track_iter():
        for pd in track:
            MAX_LONGI, MIN_LOGIN = max(pd.longitude, MAX_LONGI), min(pd.longitude, MIN_LOGIN)
            MAX_LATI, MIN_LATI = max(pd.latitude, MAX_LATI), min(pd.latitude, MIN_LATI)
            MAX_H, MIN_H = max(pd.height, MAX_H), min(pd.height, MIN_H)
    print('longitude', [MIN_LOGIN, MAX_LONGI])
    print('latitude', [MIN_LATI, MAX_LATI])
    print('height', [MIN_H, MAX_H])
