import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
X=mms.fit_transform(X)

from minisom import MiniSom 
som=MiniSom(10,10,input_len=15)
som.random_weights_init(X)
som.train_random(X,100)


from pylab import bone,pcolor,plot,show,colorbar
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']

for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None')
show()
mappings=som.win_map(X)
frauds=np.concatenate((mappings[(4,8)],mappings[(8,8)],mappings[(8,7)]),axis=0)
frauds=mms.inverse_transform(frauds)














customers=dataset.iloc[:,1:].values
is_fraud=np.zeros(690)
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1





# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Fitting classifier to the Training set
import keras
from keras.models import Sequential
from keras.layers import Dense
# Create your classifier here
classifier = Sequential()
classifier.add(Dense(units=2,activation = 'relu',kernel_initializer='uniform',input_dim=15))

classifier.add(Dense(units=1,activation = 'sigmoid',kernel_initializer='uniform',))

classifier.compile('adam',loss = 'binary_crossentropy',metrics=['accuracy'])

classifier.fit(customers,is_fraud,batch_size=1,epochs=2)



y_pred=classifier.predict(customers)
y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)


y_pred=y_pred[y_pred[:,1].argsort()]














