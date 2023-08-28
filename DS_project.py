import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn 

dataset = pd.read_csv('50_Startups.csv')
print(dataset)

print(dataset.head())
print(dataset.tail())
print(dataset.describe())
print('there are',dataset.shape[0],'rows and',dataset.shape[1],'columns in the dataset')
print('there are',dataset.duplicated().sum(),'duplicate values in the dataset')
print(dataset.isnull().sum())
print(dataset.info())

c = dataset.corr()
print(c)

sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

outliers = ['Profit']
plt.rcParams['figure.figsize']=[8,8]
sns.boxplot(data=dataset[outliers],orient='v',palette='Set2',width=0.7)
plt.title('Outliers Variables Distribution')
plt.ylabel('Profit Range')
plt.xlabel('continuous Variable')
plt.show()

sns.distplot(dataset['Profit'],bins=5, kde=True)
plt.show()

sns.pairplot(dataset)
plt.show()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)
x_train

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
('Model has been trained successfully')

y_pred = model.predict(x_test)
print(y_pred)

testing_data_model_score = model.score(x_test,y_test)
print(testing_data_model_score)

df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual value':y_test.flatten()})
print(df)

from sklearn.metrics import r2_score
r2_score = r2_score(y_pred,y_test)
print('R2 score of the Model is',r2_score)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred,y_test)
print('Mean squared error of the Model is',mse)

import numpy as np
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
('Root mean squared error of the Model is',rmse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred,y_test)
('Mean absolute error of the model is',mae)