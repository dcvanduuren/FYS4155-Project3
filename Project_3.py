import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model

df = pd.read_csv('C:.\Data_project_3\insurance.csv')#, index_col=['age'])
#print(df.head(5))
#Replace categories with numbers.
df['sex'].replace('female',0,inplace=True)
df['sex'].replace('male',1,inplace=True)
df['smoker'].replace(['no','yes'], [0,1],inplace=True)
df['region'].replace(['northwest','northeast','southeast','southwest'], [1,2,3,4],inplace=True)
#Scale the features so that all values are between 0 and 1.
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#print(df_scaled.head(5))
#Convert the pandas dataframe to a numpy array.
X_min = df_scaled[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].to_numpy()
#Add a column of numbers with value one to get a constant beta term.
x_1 = np.ones((X_min.shape[0],1))
X = np.hstack((x_1, X_min))
print(X)
y = df_scaled[['charges']].to_numpy()
#print(y)
#np.random.seed(245)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
hidden_neurons = (100, 100, 100)
batch_size = 20
epochs = 20000
eta = 0.004

#Use a neural network.
nn_reg = MLPRegressor(hidden_layer_sizes=hidden_neurons, max_iter=epochs, learning_rate_init=eta, batch_size=batch_size)
nn_reg.fit(X_train, y_train.ravel())
y_pred = nn_reg.predict(X_test)
print('R2 neural network: %g' % (r2_score(y_test, y_pred)))
print('MSE neural network: %g' % (mean_squared_error(y_test, y_pred)))

#Ordinary least squares regression
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)
y_pred_OLSreg = X_test @ beta
print('R2 ordinary least squares regression: %g' % (r2_score(y_test, y_pred_OLSreg)))
print('MSE ordinary least squares regression: %g' % (mean_squared_error(y_test, y_pred_OLSreg)))

#Ridge regression
I = np.eye(X_train.shape[1])
nlambdas = 100
lambdas = np.logspace(-6, 7, nlambdas)
r2_array = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
    y_pred_Ridge = X_test @ Ridgebeta
    r2_array[i] = r2_score(y_test, y_pred_Ridge)
    i += 1
plt.plot(np.log10(lambdas), r2_array)
plt.xlabel("log10 lamda")
plt.ylabel("R2")
plt.title("Ridge regression")
plt.show()

#Lasso regression
nlambdas = 100
lambdas = np.logspace(-6, 1, nlambdas)
r2_array = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    Lasso_reg = linear_model.Lasso(lmb)
    Lasso_reg.fit(X_train,y_train)
    y_pred_Lasso = Lasso_reg.predict(X_test)
    r2_array[i] = r2_score(y_test, y_pred_Lasso)
    i += 1
plt.plot(np.log10(lambdas), r2_array)
plt.xlabel("log10 lamda")
plt.ylabel("R2")
plt.title("Lasso regression")
plt.show()
