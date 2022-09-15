#!/usr/bin/env python
# coding: utf-8

# In[19]:


# here all library that we need
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time


# In[20]:


# to make it easy to understand what im doing here i write a definition for all the algorithm 
# that i will use them in our model so instead of write that code every time i need it i just 
# need to call the def for it and give it the X and y parameter and it will work .
def destree(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred =dtc.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("decision tree ",acc)
    print('Execution time:', elapsed_time, 'seconds')
    

def svc(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred =svc.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("svc ",acc)
    print('Execution time:', elapsed_time, 'seconds')
    

def mlp(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred =mlp.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("mlp",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
def knb(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    knb = KNeighborsClassifier()
    knb.fit(X_train, y_train)
    y_pred =knb.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("KNeighbors",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
    
def gpc(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    y_pred =gpc.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("GaussianProcess",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
def rbf(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    rbf = RBF()
    rbf.fit(X_train, y_train)
    y_pred =rbf.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("RBF",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
def rfc(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred =rfc.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("RandomForest",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
def abc(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_pred =abc.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("AdaBoost",acc)
    print('Execution time:', elapsed_time, 'seconds')
    
def gnb(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred =gnb.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("GaussianNB",acc)
    print('Execution time:', elapsed_time, 'seconds')

def gda(X,y):
    st = time.time()
    X_train, X_test, y_train, y_test =train_test_split(X, y,test_size =0.3,random_state=42)
    gda = QuadraticDiscriminantAnalysis()
    gda.fit(X_train, y_train)
    y_pred =gda.predict(X_test)
    acc = accuracy_score(y_test ,y_pred)
    et = time.time()
    elapsed_time=et-st
    print("QuadraticDiscriminantAnalysis",acc)
    print('Execution time:', elapsed_time, 'seconds')


# In[21]:


# load data by using pandas command [pd.read_csv]
data= pd.read_csv('SPAM text message.csv')
data


# ## feature extraction from text 
#  * Basics of CountVectorizer
#  * Try one without any change in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 77.1%
#      * best one was [ MLP Classifier ] with accuracy = 98.92%

# In[22]:


count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[23]:


# define df as X
X=df
X


# In[24]:


# define target column "Category" as y 
y=data['Category']
y


# In[25]:


# [1] with desision tree 
destree(X,y)


# In[26]:


# [2] with svc 
svc(X,y)


# In[27]:


# [3] with mlp
mlp(X,y)


# In[28]:


# [4] with knighbors 
knb(X,y)


# In[29]:


# [5] with gaussian process 
gpc(X,y)


# In[30]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[31]:


# [7] with random forest 
rfc(X,y)


# In[32]:


# [8] with abc
abc(X,y)


# In[33]:


# [9] with gaussianNB
gnb(X,y)


# In[34]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## Lower case
# * Try two with change the lower case to false in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 67.3%
#      * best one was [ MLP Classifier ] with accuracy = 98.86%

# In[35]:


count_vect = CountVectorizer(lowercase= False)
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[36]:


# define df as X
X=df
X


# In[37]:


# define target column "Category" as y 
y=data['Category']
y


# In[38]:


# [1] with desision tree 
destree(X,y)


# In[39]:


# [2] with svc 
svc(X,y)


# In[40]:


# [3] with mlp
mlp(X,y)


# In[41]:


# [4] with knighbors 
knb(X,y)


# In[42]:


# [5] with gaussian process 
gpc(X,y)


# In[43]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[44]:


# [7] with random forest 
rfc(X,y)


# In[45]:


# [8] with abc
abc(X,y)


# In[46]:


# [9] with gaussianNB
gnb(X,y)


# In[47]:


# [10] with quardratic discriminant analysis
gda(X,y)


# In[48]:


import nltk
# download punkt from nltk
nltk.download('punkt')


# ## stop word we choose [is , to , my ]
# * Try three with change the lower case to false in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 75.53%
#      * best one was [ MLP Classifier ] with accuracy = 98.86%

# In[49]:


count_vect = CountVectorizer(stop_words=['is','to','my'])
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[50]:


# define df as X
X=df
X


# In[51]:


# define target column "Category" as y 
y=data['Category']
y


# In[52]:


# [1] with desision tree 
destree(X,y)


# In[53]:


# [2] with svc 
svc(X,y)


# In[54]:


# [3] with mlp
mlp(X,y)


# In[55]:


# [4] with knighbors 
knb(X,y)


# In[56]:


# [5] with gaussian process 
gpc(X,y)


# In[57]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[58]:


# [7] with random forest 
rfc(X,y)


# In[59]:


# [8] with abc
abc(X,y)


# In[60]:


# [9] with gaussianNB
gnb(X,y)


# In[61]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## stop words sklearn built in stop words list 
# * Try four with change the lower case to false in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 48.86%
#      * best one was [ MLP Classifier ] with accuracy = 98.92%

# In[62]:


count_vect = CountVectorizer(stop_words='english')
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[63]:


# define df as X
X=df
X


# In[64]:


# define target column "Category" as y 
y=data['Category']
y


# In[65]:


# [1] with desision tree 
destree(X,y)


# In[66]:


# [2] with svc 
svc(X,y)


# In[67]:


# [3] with mlp
mlp(X,y)


# In[68]:


# [4] with knighbors 
knb(X,y)


# In[69]:


# [5] with gaussian process 
gpc(X,y)


# In[70]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[71]:


# [7] with random forest 
rfc(X,y)


# In[72]:


# [8] with abc
abc(X,y)


# In[73]:


# [9] with gaussianNB
gnb(X,y)


# In[74]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## CountVectorizer using max_df 
# * Try five with change the lower case to false in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 77.15%
#      * best one was [ MLP Classifier ] with accuracy = 98.92%

# In[75]:


count_vect = CountVectorizer(max_df=0.5)
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[76]:


# define df as X
X=df
X


# In[77]:


# define target column "Category" as y 
y=data['Category']
y


# In[78]:


# [1] with desision tree 
destree(X,y)


# In[79]:


# [2] with svc 
svc(X,y)


# In[80]:


# [3] with mlp
mlp(X,y)


# In[81]:


# [4] with knighbors 
knb(X,y)


# In[82]:


# [5] with gaussian process 
gpc(X,y)


# In[83]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[84]:


# [7] with random forest 
rfc(X,y)


# In[85]:


# [8] with abc
abc(X,y)


# In[86]:


# [9] with gaussianNB
gnb(X,y)


# In[87]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## CountVectorizer using min_df 
# * Try six with change the lower case to false in countvectorizer 
#      * worst one was [ Quadratic Discriminant Analysis ] with accuracy = 69.79%
#      * best one was [ Random forest ] with accuracy = 98.08%

# In[88]:


count_vect = CountVectorizer(min_df=0.01)
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[89]:


# define df as X
X=df
X


# In[90]:


# define target column "Category" as y 
y=data['Category']
y


# In[91]:


# [1] with desision tree 
destree(X,y)


# In[92]:


# [2] with svc 
svc(X,y)


# In[93]:


# [3] with mlp
mlp(X,y)


# In[94]:


# [4] with knighbors 
knb(X,y)


# In[95]:


# [5] with gaussian process 
gpc(X,y)


# In[96]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[97]:


# [7] with random forest 
rfc(X,y)


# In[98]:


# [8] with abc
abc(X,y)


# In[99]:


# [9] with gaussianNB
gnb(X,y)


# In[100]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## CountVectorizer using max_feature 
# * Try seven with change the lower case to false in countvectorizer 
#      * worst one was [ GaussianNB ] with accuracy = 80.80%
#      * best one was [ MLP ] with accuracy = 98.92%

# In[101]:


count_vect = CountVectorizer(max_features=1000)
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[102]:


# define df as X
X=df
X


# In[103]:


# define target column "Category" as y 
y=data['Category']
y


# In[104]:


# [1] with desision tree 
destree(X,y)


# In[105]:


# [2] with svc 
svc(X,y)


# In[106]:


# [3] with mlp
mlp(X,y)


# In[107]:


# [4] with knighbors 
knb(X,y)


# In[108]:


# [5] with gaussian process 
gpc(X,y)


# In[109]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[110]:


# [7] with random forest 
rfc(X,y)


# In[111]:


# [8] with abc
abc(X,y)


# In[112]:


# [9] with gaussianNB
gnb(X,y)


# In[113]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## TfidVectorizer : Feature Generation using TF-IDF
# * Try eight with change the lower case to false in countvectorizer 
#      * worst one was [ GaussianNB ] with accuracy = 80.80%
#      * best one was [ MLP ] with accuracy = 98.92%

# In[114]:


from sklearn.feature_extraction.text import TfidfVectorizer
# create the transform

vectorizer = TfidfVectorizer()
matrix =vectorizer.fit_transform(data.Message)
array = matrix.toarray()
new_data = pd.DataFrame(data=array,columns=vectorizer.get_feature_names())
new_data


# In[115]:


# define df as X
X=df
X


# In[116]:


# define target column "Category" as y 
y=data['Category']
y


# In[117]:


# [1] with desision tree 
destree(X,y)


# In[118]:


# [2] with svc 
svc(X,y)


# In[119]:


# [3] with mlp
mlp(X,y)


# In[120]:


# [4] with knighbors 
knb(X,y)


# In[121]:


# [5] with gaussian process 
gpc(X,y)


# In[122]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[123]:


# [7] with random forest 
rfc(X,y)


# In[124]:


# [8] with abc
abc(X,y)


# In[125]:


# [9] with gaussianNB
gnb(X,y)


# In[126]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## here i will combine some method to gather   

# # stop_word with max_df
# * Try nine with change the lower case to false in countvectorizer 
#      * worst one was [ QuadraticDiscriminantAnalysis] with accuracy = 48%
#      * best one was [ MLP ] with accuracy = 98.92%

# In[127]:


coun_vect = CountVectorizer(stop_words= 'english',max_df=0.65)
count_matrix = coun_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
df


# In[128]:


# define df as X
X=df
X


# In[129]:


# define target column "Category" as y 
y=data['Category']
y


# In[130]:


# [1] with desision tree 
destree(X,y)


# In[131]:


# [2] with svc 
svc(X,y)


# In[132]:


# [3] with mlp
mlp(X,y)


# In[133]:


# [4] with knighbors 
knb(X,y)


# In[134]:


# [5] with gaussian process 
gpc(X,y)


# In[135]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[136]:


# [7] with random forest 
rfc(X,y)


# In[137]:


# [8] with abc
abc(X,y)


# In[138]:


# [9] with gaussianNB
gnb(X,y)


# In[139]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ##  stop word with min_df 
# * Try ten with change the lower case to false in countvectorizer 
#      * worst one was [ GaussianNB] with accuracy = 53.16%
#      * best one was [ MLP ] with accuracy = 98.92%
# 

# In[140]:


coun_vect = CountVectorizer(stop_words= 'english',min_df=0.01)
count_matrix = coun_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
df


# In[141]:


# define df as X
X=df
X


# In[142]:


# define target column "Category" as y 
y=data['Category']
y


# In[143]:


# [1] with desision tree 
destree(X,y)


# In[144]:


# [2] with svc 
svc(X,y)


# In[145]:


# [3] with mlp
mlp(X,y)


# In[146]:


# [4] with knighbors 
knb(X,y)


# In[147]:


# [5] with gaussian process 
gpc(X,y)


# In[148]:


# [6] with RBF
#rbf(X,y)
#AttributeError: 'RBF' object has no attribute 'fit'


# In[149]:


# [7] with random forest 
rfc(X,y)


# In[150]:


# [8] with abc
abc(X,y)


# In[151]:


# [9] with gaussianNB
gnb(X,y)


# In[152]:


# [10] with quardratic discriminant analysis
gda(X,y)


# ## here i will try one of the best result in loop

# In[153]:


count_vect = CountVectorizer(stop_words='english')
count_matrix = count_vect.fit_transform(data.Message)
count_array = count_matrix.toarray()
df = pd.DataFrame(data= count_array,columns=count_vect.get_feature_names())
df


# In[154]:


import numpy as np
X = df
y = data['Category']




acc_list = []
N = 20
st = time.time()
for i in range(N):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    MLP = MLPClassifier()
    MLP.fit(X_train, y_train)
    y_pred = MLP.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    print(acc)
    
et = time.time()
elapsed_time=et-st
np.mean(acc_list)
print('Execution time:', elapsed_time, 'seconds')


# In[ ]:




