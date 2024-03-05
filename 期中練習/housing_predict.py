import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("Housing_Dataset_Sample.csv")

# df.head()
# df.describe().T
# sns.displot(df['Price'])
# sns.jointplot(df['Avg. Area Income'], df['Price'])
# sns.pairplot(df)

X = df. iloc[:,:5]
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 54)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

from sklearn.metrics import r2_score
#作答紀錄跟標準答案對照
r2_score(y_test, predictions)

plt.scatter(y_test, predictions, color = 'blue', alpha=0.3)