#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#load dataset
data=pd.read_csv("C:\\Users\\54721\\OneDrive\\Desktop\\kagle dataset\\linear regression dataset\\train.csv")

#quick view about the dataset
print(data)

#Return a tuple representing the dimensionality of the DataFrame.
print(data.shape)

#data.plot(kind='scatter',x="x",y="y")
# plt.show()

#data.plot(kind='box')
# plt.show()

#correlation coeffecients
#print(data.corr())

#change to dataframe variable
X_axis=pd.DataFrame(data['x'])
print(X_axis)
Y_axis=pd.DataFrame(data['y'])
print(Y_axis)

#build linear regression model
lm = linear_model.LinearRegression()
model = lm.fit(X_axis,Y_axis)

print(model.predict(X_axis))
print(model.coef_)
print(model.intercept_)

#evaluate the model
print(model.score(X_axis,Y_axis))

#predict new value of Y
X_axis_new = [[24]]
Y_axis_predict = model.predict(X_axis_new)
print(Y_axis_predict)

#predict more values
a= [6,78,91]

a=pd.DataFrame(a)
print(a)

#change the column name from 0 to any string value otherwise keyerror occur when plotting graph.
a.columns = ['x']
print(a)

b=model.predict(a)
print(b)

df=pd.DataFrame(b)
print(df)

#visualize the results

data.plot(kind='scatter',x= 'x' ,y='y')

#plotting the regression line
plt.plot(X_axis,model.predict(X_axis),color='red',linewidth=2)

#plotting the predicted value for X_axis_new = [[24]]
plt.scatter(X_axis_new,Y_axis_predict,color='yellow')

#plotting the predicted value for sample a= [6,78,91]
plt.scatter(a,b,color='green',linewidth=3)

#blue line connecting the new values of y for new sample data a= [6,78,91]
plt.plot(a,b,color='blue',linewidth=3)


#show the graph
plt.show()
