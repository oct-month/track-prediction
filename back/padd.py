import pandas as pd

INTERVAL = 500

dataframe = pd.read_table('./data/20210401.txt', sep='\t')

df = dataframe.loc[dataframe['航班号(I170)'] == 'CES2770 ']

# df = dataframe.loc[range(0, dataframe.shape[0], INTERVAL)]



df.to_csv('data.csv', index=False, encoding='GBK')
