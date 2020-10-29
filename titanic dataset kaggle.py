#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pwd


# In[3]:


df=pd.read_csv('C:\\Users\\Ashutosh\\Downloads\\train.csv')


# In[4]:


df.head()


# In[5]:


df.isnull()


# In[6]:


sns.heatmap(df.isnull())


# In[7]:


sns.countplot(df.Survived,hue=df.Sex)


# In[38]:


sns.jointplot(df.Survived,df.Age)


# In[43]:


plt.scatter(df.Survived,df.Age)


# In[46]:


plt.hist(df.SibSp)


# In[50]:


sns.countplot(df.SibSp,hue=df.Survived)


# In[54]:


df.plot('Fare',kind='hist')


# In[56]:


plt.hist(df.Fare,bins=40)


# In[58]:


plt.scatter(df.Survived,df.Fare)


# In[60]:


sns.jointplot(df.Survived,df.Fare)


# In[5]:


l=df.Age.mean()
df.Age=df.Age.fillna(value=l)
ls=list(df.Age)


# In[6]:


df.drop(['Cabin'],axis=1,inplace=True)


# In[7]:


df.head()


# In[130]:


sns.heatmap(df.isnull())


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


lbl=LabelEncoder()


# In[10]:


df.iloc[:,4]=lbl.fit_transform(df.iloc[:,4])
df.Embarked=df.fillna(value='S')
df.iloc[:,-1]=lbl.fit_transform(df.iloc[:,-1])
df


# In[33]:


#df.drop(['Fare'],axis=1,inplace=True)
x=df.iloc[:,1:]

#df.drop(['PassengerId'],axis=1,inplace=True)
#df.drop(['Embarked','Age'],axis=1,inplace=True)
#df.drop(['Parch'],axis=1,inplace=True)
#df.drop(['SibSp'],axis=1,inplace=True)
x


# In[34]:


y=df.iloc[:,0]
#df.drop(['Name','Ticket'],axis=1,inplace=True)
y


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[35]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[36]:


lgt=LogisticRegression()


# In[37]:


lgt.fit(xtrain,ytrain);


# In[38]:


pred=lgt.predict(xtest)
nls=list(pred)


# In[39]:


metrics.accuracy_score(ytest,pred)


# In[20]:


# Here we are importing test data to predict
dd=pd.read_csv('C:\\Users\\Ashutosh\\Downloads\\test.csv')


# In[21]:


dd


# In[38]:


sns.heatmap(dd.isnull())


# In[22]:


sl=dd.Age.mean()


# In[23]:


dd.Age=dd.Age.fillna(value=sl)


# In[24]:


sns.heatmap(dd.isnull())


# In[43]:


sns.heatmap(dd.isnull())


# In[34]:


#dd.drop(['Name','Cabin','Ticket','Fare'],axis=1,inplace=True)
#dd.drop(['PassengerId'],axis=1,inplace=True)
#dd.drop(['Embarked'],axis=1,inplace=True)
dd


# In[35]:


dd.iloc[:,1]=lbl.fit_transform(dd.iloc[:,1])

dd


# In[53]:


xd=dd.iloc[:,:]

xd.Parch=x.Parch.fillna(x.Parch.mean())
xd.SibSp=x.SibSp.fillna(x.SibSp.mean())
xd.Age=x.Age.fillna(x.Age.mean())


# In[54]:


spred=lgt.predict(xd)


# In[62]:


spred


# In[63]:


sns.countplot(dd.Pclass,hue=dd.Sex)


# In[53]:


pwd


# In[54]:


pd.read_csv('C:\\Users\\Ashutosh\\Downloads\\gender_submission.csv')


# In[69]:


Survived=pd.DataFrame({'Survived':spred})
Survived


# In[68]:


PassengerId=pd.DataFrame({'PassengerId':range(892,1310)})
PassengerId


# In[71]:


df=pd.concat([PassengerId,Survived],axis=1)


# In[76]:


df.to_csv('C:\\Users\\Ashutosh\\Downloads\\Results.csv',index=False)


# In[77]:


pd.read_csv('C:\\Users\\Ashutosh\\Results.csv')


# In[ ]:




