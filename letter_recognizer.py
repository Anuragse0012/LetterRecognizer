import numpy as np
import os
import matplotlib.pyplot as pt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics,svm
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


data = pd.read_csv(os.path.join(os.getcwd(),'train.csv')).as_matrix()
#knn = KNeighborsClassifier(n_neighbors=)
#svm = svm.SVC(gamma=0.001, C=100., kernel = 'linear')
knn = DecisionTreeClassifier()
features = ['x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']

xtrain = data[0:15000,1:]
train_label = data[0:15000,0]

#print(train_label)
print(data[0])
print(xtrain.shape) # (16000,17)
print(train_label.shape) #(16000,)

knn.fit(xtrain,train_label)

#print(xtrain)
#sprint(train_label)



# testing data
test_data = pd.read_csv(os.path.join(os.getcwd(),'test.csv')).as_matrix()

#xtest = data[15000:,1:]
#ytest = data[15000:,0]
#print(type(ytest))
xtest = test_data[0:len(test_data),0:]
#print(xtest)

y_pred = knn.predict(xtest)

print('Prediction')
print(type(y_pred))

for row in y_pred:
    for cell in np.nditer(row):
        print(cell, end=' ')

print('\n')
#print('Accuracy:',metrics.accuracy_score(ytest,y_pred))

'''test_letter = xtest[6]
print(xtest[6].shape)

pt.imshow(test_letter.shape, cmap='gray')
print(knn.predict([xtest[6]]))
pt.show()'''
