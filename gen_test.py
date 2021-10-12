from collections import defaultdict

from data_loader import PlaneData


if __name__ == '__main__':
    data_sets = defaultdict(list)

    max_height, min_height = -1.0, 100000.0

    with open('data/20210401.txt', 'r', encoding='UTF-8') as f:
        t = f.readlines()[1:]
        for s in t:
            pd = PlaneData.from_str(s)

            if pd.is_available:
                data_sets[pd.flight_number].append(s)

                max_height = max(pd.height, max_height)
                min_height = min(pd.height, min_height)

            # if pd.height < 2000 and pd.is_available:
            #     steps.add(pd.flight_number)
            
            # if len(steps) == 4 and len(result) == 0:
            #     result.append(s)
            # elif len(result) > 0 and pd.flight_number == PlaneData.from_str(result[-1]).flight_number:
            #     result.append(s)

    # for k, v in data_sets.items():
    #     if PlaneData.from_str(v[0]).height - PlaneData.from_str(v[-1]).height > 2000 and PlaneData.from_str(v[-1]).height < 5000:
    #         result = v
    #         break

    print(max_height, min_height)

    for fn, dts in data_sets.items():
        with open(f'test/{fn}.txt', 'w', encoding='UTF-8') as f:
            f.writelines(dts)
