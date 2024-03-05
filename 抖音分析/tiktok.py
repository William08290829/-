import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("tiktok.csv")

df.head()
df.info()
df['track_name'].value_counts()
df['artist_name'].value_counts()

# genre 類型
sns.distplot(df['genre'])

# drop
df.drop(['track_id','artist_id'],axis=1,inplace=True)
df.head()
