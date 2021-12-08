# import
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# dataframe
df = pd.read_csv('../Project 3/FYS4155-Project3/insurance.csv')

#Replace categories with numbers.
df['sex'].replace('female',0,inplace=True)
df['sex'].replace('male',1,inplace=True)
df['smoker'].replace(['no','yes'], [0,1],inplace=True)
df['region'].replace(['northwest','northeast','southeast','southwest'], [1,2,3,4],inplace=True)
#Scale the features so that all values are between 0 and 1.
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# correlation plot
corr = df_scaled.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True)


#import sklearn LinReg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

X = df.drop(['charges'], axis = 1)
Y = df.charges

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)
LinReg = LinearRegression().fit(X_train,Y_train)

y_train_pred = LinReg.predict(X_train)
y_test_pred = LinReg.predict(X_test)

print(LinReg.score(X_test,Y_test))


#RFR
RFR = RandomForestRegressor(n_estimators = 100,criterion = 'mse',random_state = 1, n_jobs = -1)
RFR.fit(X_train,Y_train)
RFR_train_pred = RFR.predict(X_train)
RFR_test_pred = RFR.predict(X_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (mean_squared_error(Y_train,RFR_train_pred),mean_squared_error(Y_test,RFR_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (r2_score(Y_train,RFR_train_pred),r2_score(Y_test,RFR_test_pred)))

plt.figure(figsize=(10,8))

plt.scatter(RFR_train_pred,RFR_train_pred - Y_train, c = 'c', marker = 'o', s = 20, alpha = 0.8,label = 'Train data')
plt.scatter(RFR_test_pred,RFR_test_pred - Y_test, c = 'purple', marker = 'o', s = 20, alpha = 0.8,label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper right')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'black',)
plt.show()