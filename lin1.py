# import
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# dataframe
df = pd.read_csv('../Project 3/FYS4155-Project3/insurance.csv')

#sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)

# correlation plot
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True)
#plt.show()


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
