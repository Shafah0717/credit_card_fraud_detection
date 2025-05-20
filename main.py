import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("creditcard.csv")
print(data.head())
print(data['Class'].value_counts())

x = data.drop('Class',axis=1) #to features
y = data['Class'] #to label

x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
