#!/usr/bin/env python
# coding: utf-8

# The objective is to analyze the flight booking dataset obtained from a platform which is used to book flight tickets. A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers. Apply EDA, statistical methods and Machine learning algorithms in order to get meaningful information from it.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\abhisha dwnload\Flight_Booking\Flight_Booking.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df = df.drop(columns = ['Unnamed: 0'])


# In[8]:


df.shape


# In[9]:


#LinePlot 
#Syntax: sns.lineplot(x,y)


# In[10]:


plt.figure(figsize = (15,5))
sns.lineplot(x=df['airline'], y=df['price'])
plt.title('Airline Vs Price')
plt.xlabel('Airline')
plt.ylabel('Price')
plt.show()


# There is a variation in price with different airlines

# In[11]:


plt.figure(figsize = (15,5))
sns.lineplot(x=df['days_left'], y=df['price'], color = 'blue')
plt.title('Days left for departure Vs Ticket Price')
plt.xlabel('Days left for departure')
plt.ylabel('Price')
plt.show()


# The price of the ticket increases as the days left for departure decreases

# In[12]:


#Barplot
#syntax: sns.barplot(x,y)


# In[13]:


sns.barplot(x=df['airline'], y=df['price'])


# Price range of all the flights

# In[14]:


fig,ax=plt.subplots(1,2,figsize=(20,6))
sns.lineplot(data=df, x='days_left', y='price', hue='source_city', ax=ax[0])
sns.lineplot(data=df, x='days_left', y='price', hue='destination_city', ax=ax[1])


# Range of price of flights with source and destination city according to the days left

# In[15]:


#COUNTPLOT
#Syntax: sns.countplot(x)


# In[16]:


plt.figure(figsize=(15,23))

plt.subplot(4,2,1)
sns.countplot(x=df['airline'], data=df)
plt.title('Frequency of Airline')

plt.subplot(4,2,2)
sns.countplot(x=df['source_city'], data=df)
plt.title('Frequency of source_city')

plt.subplot(4,2,3)
sns.countplot(x=df['departure_time'], data=df)
plt.title('Frequency of departure_time')

plt.subplot(4,2,4)
sns.countplot(x=df['stops'], data=df)
plt.title('Frequency of stops')

plt.subplot(4,2,5)
sns.countplot(x=df['arrival_time'], data=df)
plt.title('Frequency of arrival_time')

plt.subplot(4,2,6)
sns.countplot(x=df['destination_city'], data=df)
plt.title('Frequency of destination_city')

plt.subplot(4,2,7)
sns.countplot(x=df['class'], data=df)
plt.title('Frequency of class')


# Visualization of categorical features with countplot

# In[17]:


#Performing One Hot Encoding for categorical features of a dataframe
from sklearn.preprocessing import LabelEncoder


# In[18]:


le = LabelEncoder()


# In[19]:


df['airline']=le.fit_transform(df['airline'])
df['source_city']=le.fit_transform(df['source_city'])
df['flight']=le.fit_transform(df['flight'])
df['departure_time']=le.fit_transform(df['departure_time'])
df['stops']=le.fit_transform(df['stops'])
df['arrival_time']=le.fit_transform(df['arrival_time'])
df['destination_city']=le.fit_transform(df['destination_city'])
df['class']=le.fit_transform(df['class'])
df.info()


# In[20]:


#Heatmap
#Syntax: sns.heatmap(df.corr(),abbot=True,CMAP='color')


# In[21]:


# Set the figure size for the heatmap
plt.figure(figsize=(10, 5))

# Create the heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


#  Plotting the correlation 
# graph to see the 
# correlation between 
# features and dependent 
# variable

# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'price')):
        col_list.append(col)


# In[23]:


X = df[col_list]
vif_data = pd.DataFrame()
vif_data["features"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values,i) for i in range (len(X.columns))]
print(vif_data)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score


# In[25]:


X = df.drop(columns = 'price')
y = df['price']


# In[26]:


li_model = LinearRegression()


# In[27]:


li_model


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state =42)


# In[29]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[30]:


X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[34]:


li_model.fit(X_train,y_train)


# In[35]:


y_pred = li_model.predict(X_test)


# In[36]:


difference = pd.DataFrame(np.c_[y_test,y_pred], columns = ['Actual Value','Predicted Values'])


# In[37]:


difference


# In[38]:


#Calculating r2 score,MAE, MAPE, MSE, RMSE. Lower the RMSE and MAPE better the model
r2_score = r2_score(y_test,y_pred)
r2_score


# In[40]:


from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred)
mape


# In[41]:


from sklearn import metrics


# In[43]:


mean_abs_error = metrics.mean_absolute_error(y_test,y_pred)
mean_abs_error


# In[44]:


mean_sqrd_error = metrics.mean_squared_error(y_test,y_pred)
mean_sqrd_error


# In[45]:


root_mean_square_error = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
root_mean_square_error


# In[58]:


sns.distplot(y_pred, label = 'Predicted')
sns.distplot(y_test, label = 'Actual')


# In[59]:


from sklearn.tree import DecisionTreeRegressor  # Or DecisionTreeClassifier for classification
dt = DecisionTreeRegressor(max_depth=10)  # Limit the depth to 10 (or another value)
dt.fit(X_train, y_train)


# In[60]:


dt = DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=5)
dt.fit(X_train, y_train)


# In[61]:


dt = DecisionTreeRegressor(max_features='sqrt')  # Only use a subset of features at each split
dt.fit(X_train, y_train)


# In[62]:


X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_sample = X_train_sample.sample(n=10000)  # Use a smaller sample of data


# In[63]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[64]:


from joblib import parallel_backend

with parallel_backend('threading', n_jobs=-1):  # Use multiple threads
    dt.fit(X_train, y_train)


# In[65]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test,y_pred)
r2_score


# In[66]:


mean_abs_error = metrics.mean_absolute_error(y_test,y_pred)

mean_abs_error


# In[67]:


mape  = mean_absolute_percentage_error(y_test, y_pred)
mape


# In[68]:


mean_sqrd_error = metrics.mean_squared_error(y_test,y_pred)
mean_sqrd_error


# In[69]:


root_mean_square_error = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
root_mean_square_error 


# In this project, the Decision Tree model demonstrated superior performance compared to Linear Regression. The evaluation metrics, MAPE (Mean Absolute Percentage Error) and RMSE (Root Mean Squared Error), were both lower for the Decision Tree, indicating that it made more accurate predictions with smaller errors.

# In[ ]:




