DATA_DIR_1 = './datasets/radar'
DATA_DIR_2 = './datasets/cache'
DATA_DIR_3 = './datasets/train'
DATA_DIR_4 = './datasets/load'


TRACK_MIN_POINT_NUM = 50                # 一条航迹最少应有50航迹点

FEATURES_COLUMNS = ['时间', '经度', '纬度', '速度', '高度', '航向']
LABEL_COLUMNS = ['时间', '经度', '纬度', '高度']


FEATURES_NORMALIZATION = [
    [1000000000, 4000000000],
    [-180, 180],
    [-90, 90],
    [0, 5000],
    [-10000, 20000],
    [0, 360]
]
LABEL_NORMALIZATION = [
    FEATURES_NORMALIZATION[0],
    FEATURES_NORMALIZATION[1],
    FEATURES_NORMALIZATION[2],
    FEATURES_NORMALIZATION[4]
]
NORMALIZATION_TIMES = 100

# LSTM seq_length
SEQ_LENGTH = 6
# 每纬度大约多少m
PER_LATITUDE_M = 111319.48598161776
# 每经度大约多少m
PER_LONGITUDE_M = (87695.1394386052 + 91939.33948001463) / 2
# 基准纬度
BASE_LATITUDE = 34
# 基准经度
BASE_LONGITUDE = 108

batch_size = 256
num_epochs = 5
lr = 1000

PARAMS_PATH = './params-hybrid.pt'
