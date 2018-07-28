from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import numpy as np 

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True) #no relation so remove it

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy) 

example_measures = np.array([4,2,1,1,1,3,2,1]) #we'll test on this
example_measures = example_measures.reshape(1, -1) #if not done this then gives error as it looks even for id

prediction = clf.predict(example_measures)

print(prediction)