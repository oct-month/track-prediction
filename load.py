from typing import List

from data import PlaneData
from config import DATA_SOURCE

def load_data() -> List[PlaneData]:
    result = []
    with open(DATA_SOURCE, 'r', encoding='UTF-8') as f:
        for s in f.readlines()[1:]:
            pd = PlaneData.from_str(s)
            if pd.is_available:
                result.append(pd)
    return result
