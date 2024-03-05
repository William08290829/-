import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn import preprocessing

import pydotplus
from IPython.display import Image

launch_data = pd.read_excel("RocketLaunchDataCompleted.xlsx")
# launch_data.info()
# launch_data.head()
# launch_data.count()

launch_data.isnull().sum    #計算NaN的出現次數

#沒有資料代表沒有發射
launch_data['Launched?'].value_counts()
launch_data['Launched?'].fillna('N', inplace=True)
launch_data['Launched?'].value_counts()

#沒有資料的補上"沒有太空人"
launch_data['Crewed or Uncrewed'].value_counts()
launch_data['Crewed or Uncrewed'].fillna('Uncrewed', inplace=True)
launch_data['Crewed or Uncrewed'].value_counts()

#天氣
launch_data['Condition'].value_counts() #有兩筆資料沒有數據
launch_data['Condition'].isnull().sum()
launch_data['Condition'].fillna('Cloudy',inplace=True)

#風向
launch_data['Wind Direction'].value_counts()
launch_data['Wind Direction'].isnull().sum()    #有一個
launch_data['Wind Direction'].fillna('unknown',inplace=True)    #補上unknown

#其餘數值類的，全部補上0
launch_data.isnull().sum()
launch_data.fillna(0, inplace=True)


launch_data.info()
label_encoder = preprocessing.LabelEncoder()
# 把這三個資料換成數字
launch_data['Crewed or Uncrewed'].value_counts()
launch_data['Wind Direction'].value_counts()
launch_data['Condition'].value_counts()
launch_data['Crewed or Uncrewed'] = label_encoder.fit_transform(launch_data['Crewed or Uncrewed'])
launch_data['Wind Direction'] = label_encoder.fit_transform(launch_data['Wind Direction'])
launch_data['Condition'] = label_encoder.fit_transform(launch_data['Condition'])

y = launch_data['Launched?']
# Removing the columns we are not interested in
X = launch_data.drop(['Name','Date','Time (East Coast)','Location','Launched?', 'Hist Ave Sea Level Pressure','Sea Level Pressure','Day Length','Notes','Hist Ave Visibility', 'Hist Ave Max Wind Speed'],axis=1)
X.info()


#模型
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99)
tree_model.fit(X_train, y_train)
predictions = tree_model.predict(X_test)
predictions

#兩個相等
print(metrics.accuracy_score(y_test, predictions))

print(tree_model.score(X_test, y_test))

from sklearn.tree import export_graphviz
tree_str = export_graphviz(tree_model, feature_names=X.columns.values, class_names=['No Launch', 'Launch'],filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(tree_str)
Image(graph.create_png())

#單筆測試
X.info()
data_input = [1, 75.0, 68.0, 71.0, 0.0, 75.0, 55.0, 65.0, 0.0, 0.08, 0, 16.0, 15.0, 0.0, 0]
tree_model.predict([data_input])