import pandas as pd
DF = pd.read_parquet('Data/DF.parquet')
df = DF.sample(n=60000)
df.to_parquet('Data/Acc/df.parquet')
