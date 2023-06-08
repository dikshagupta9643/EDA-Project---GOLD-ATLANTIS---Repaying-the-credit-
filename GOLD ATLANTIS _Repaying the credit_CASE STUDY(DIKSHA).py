#!/usr/bin/env python
# coding: utf-8

# # GOLD ATLANTIS : Repaying the credit

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sb


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv("DS1_C5_S4_Credit_Datas_Hackathon.csv")
df


# # Level 0 Analysis: Understanding Data

# In[5]:


# reading first 5 rows
df.head()


# In[6]:


# reading the last 5 rows
df.tail()


# In[7]:


# number of rows and columns
df.shape


# In[8]:


# name of all columns
df.columns


# In[9]:


# information of dataset
df.info()


# In[10]:


# checking the null values
df.isnull().sum()


# In[11]:


# creating a heat map
fig, ax = plt.subplots(figsize = (15,7))
sb.heatmap(df.corr(), cbar = True, linewidths = 0.5, annot = True)
plt.show()


# # LEVEL 1 Analysis

# In[12]:


# separating nummerical and categorical data types columns
def sep_data_types(df):
    numerical = []
    categorical = []
    for col in df.columns:
        if df[col].nunique() < 100:
            categorical.append(col)
        else :
            numerical.append(col)
    return categorical, numerical

categorical, numerical = sep_data_types(df)
from tabulate import tabulate
table = [categorical, numerical]
print(tabulate({"categorical" : categorical, 'numerical' :  numerical}, headers = ['categorical', 'numerical']))


# In[13]:


# creating function for categorical columns
def info_of_cat(col):
    print(f" unique values in {col} are: {df[col].unique()}")
    print(f" mode of {col} are: {df[col].mode()[0]}")
    print(f" total missing values in {col} are: {df[col].isnull().sum()}")


# In[14]:


info_of_cat('TARGET')


# In[15]:


info_of_cat('NAME_CONTRACT_TYPE')


# In[16]:


info_of_cat('GENDER')


# In[17]:


info_of_cat('Car')


# In[18]:


info_of_cat('House')


# In[19]:


info_of_cat('CNT_CHILDREN')


# In[20]:


info_of_cat('NAME_TYPE_SUITE')


# In[21]:


# imputing the missing value in NAME_TYPE_SUITE column
df['NAME_TYPE_SUITE'].fillna('Unaccompanied', inplace = True)


# In[22]:


info_of_cat('NAME_TYPE_SUITE')


# In[23]:


info_of_cat('NAME_INCOME_TYPE')


# In[24]:


info_of_cat('NAME_EDUCATION_TYPE')


# In[25]:


info_of_cat('NAME_FAMILY_STATUS')


# In[26]:


info_of_cat('MOBILE')


# In[27]:


info_of_cat('OCCUPATION_TYPE')


# In[28]:


df['OCCUPATION_TYPE'].fillna('Laborers', inplace = True)


# In[29]:


info_of_cat('OCCUPATION_TYPE')


# In[30]:


info_of_cat('CNT_FAM_MEMBERS')


# In[31]:


df['CNT_FAM_MEMBERS'].fillna('2', inplace = True)


# In[32]:


info_of_cat('CNT_FAM_MEMBERS')


# In[33]:


info_of_cat('APPLICATION_DAY')


# In[34]:


info_of_cat('TOTAL_DOC_SUBMITTED')


# In[35]:


# defininig a function for numerical columns
def info_of_num(col):
    print (f"mean of the {col} is : {df[col].mean()}")
    print(f"median of the {col} is : {df[col].median()}")
    print(f"mode of {col} is : {df[col].mode()[0]}")
    print(f" standard deviation of {col} is : {df[col].std()}")
    print(f"total missing values in {col} is : {df[col].isnull().sum()}")
    


# In[36]:


info_of_num('SK_ID_CURR')


# In[37]:


info_of_num('AMT_INCOME_TOTAL')


# In[38]:


info_of_num('AMT_CREDIT')


# In[39]:


info_of_num('AMT_GOODS_PRICE')


# In[40]:


df['AMT_GOODS_PRICE'].fillna(450000, inplace = True)


# In[41]:


info_of_num('AMT_GOODS_PRICE')


# In[42]:


info_of_num('DAYS_EMPLOYED')


# # Analysis of number of children client has
# 

# In[43]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['CNT_CHILDREN'])
plt.title('count of clients having children')
plt.show()


# ## INTERPRETATION :
# * maximum clients do not have children.

# # Analysing the clients which provide work phone

# In[44]:


fig, ax = plt.subplots(figsize = (15,7))
data = df['WORK_PHONE'].value_counts()
labels = data.keys()
plt.pie(x = data, labels = labels, autopct = "%0.2f%%")
plt.title('percenatge of employees who provide work phone')
plt.show()


# ## INTERPRETATION :
# * 82.13 % clients provide their work phone.

# # Analysing the clients which provide home phone

# In[45]:


fig, ax = plt.subplots(figsize = (15,7))
data = df['HOME_PHONE'].value_counts()
labels = data.keys()
plt.pie(x = data, labels = labels, autopct = "%0.2f%%")
plt.title('percenatge of employees who provide home phone')
plt.show()


# ## INTERPRETATION :
# * 20% clients provide the home phone.

# # analysing count of family members of client
# 

# In[46]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['CNT_FAM_MEMBERS'])
plt.title('count of family members of client')
plt.show()


# In[49]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['NAME_EDUCATION_TYPE'])
#plt.title('count of family members of client')
plt.show()


# In[48]:


df.columns


# ## INTERPRETATION:
# * Maximum clients have 2 family members

# # Aanlysing the number of documents submitted by clients

# In[46]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TOTAL_DOC_SUBMITTED'])
plt.title('number of documents submitted by client')
plt.show()


# ## INTERPRETATION :
# * maximum clients have submitted only 1 document.

# # analysing the target variable

# In[47]:


fig, ax = plt.subplots(figsize = (15,7))
data = df['TARGET'].value_counts()
labels = data.keys()
plt.pie(x = data, labels = labels, autopct = "%0.2f%%")
plt.title('target variable')
plt.show()


# ## INTERPRETATION :
# * 8% clients have difficulties with payment.

# # analysing the number of days before the application the person started current employment

# In[48]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(x = 'DAYS_EMPLOYED', data = df, color = 'g', bins = [2500,5000,7500,10000,12500,15000,17500, 20000])
plt.title('count of days before the application client started current employment ')
plt.show()


# ## INTERPRETATION :
# * Maximum employees applied for loan after 2500 to 5000 days of starting the current employment. 

# # Analysing the credit amount of loan

# In[49]:


fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.histplot(x = 'AMT_CREDIT', data = df, ax=ax[0], color = 'r', bins = [0,500000,1000000,1500000,2000000, 2500000])
sb.boxplot(x = 'AMT_CREDIT', data = df, ax=ax[1], color = 'b')
plt.show()


# ## INTERPRETATION :
# * Maximum employees have the credit amount less than 5 lacs.
# * there are certain outliers which needs to be treated and some outliers are showing a specific pattern.

# In[51]:


mean = df['AMT_CREDIT'].mean()
print(mean)


# In[52]:


x = df[df['AMT_CREDIT'] > 2500000].index
print(x)


# In[53]:


for index in x :
    df.loc[index, 'AMT_CREDIT'] = 599003.4465


# In[54]:


fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.histplot(x = 'AMT_CREDIT', data = df, ax=ax[0], color = 'r', bins = [500000,1000000,1500000,2000000, 2500000])
sb.boxplot(x = 'AMT_CREDIT', data = df, ax=ax[1], color = 'b')
plt.show()


# ## INTERPRETATION :
# * maximum clients have the credit amount less than 10 lakhs.
# * there are outliers which are showing. a specific pattern .

# # Analysing the good price for consumer loans

# In[55]:


fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.histplot(x = 'AMT_GOODS_PRICE', data = df, ax=ax[0], color = 'orange', bins = 8)
sb.boxplot(x = 'AMT_GOODS_PRICE', data = df, ax=ax[1], color = 'purple')
plt.show()


# ## INTERPRETATION :
# * Maximum clients have good price less than 5 lakhs for consumer loans.
# * there are certain outliers which needs to be treated.

# In[56]:


# treating the outliers
mean = df['AMT_GOODS_PRICE'].mean()
print(mean)


# In[57]:


x = df[df['AMT_GOODS_PRICE'] > 2300000].index
print(x)


# In[60]:


for index in x:
    df.loc[index, 'AMT_GOODS_PRICE'] = 538273.5894


# In[61]:


# after outlier treatment
fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.histplot(x = 'AMT_GOODS_PRICE', data = df, ax=ax[0], color = 'orange', bins = 5)
sb.boxplot(x = 'AMT_GOODS_PRICE', data = df, ax=ax[1], color = 'purple')
plt.show()


# ## INTERPRETATION :
# * Maximum clients have good price less than 5 lakhs for consumer loans.
# * there are certain outliers which are showing a specific pattern.

# # Analysing the income of the client

# In[65]:


fig, ax = plt.subplots( figsize = (15,7))
sb.histplot(x = 'AMT_INCOME_TOTAL', data = df,  color = 'm', bins = [1000000, 2000000,3000000,4000000,5000000,6000000,7000000,8000000])
plt.show()


# ## INTERPRETATION :
# * maximum clients have income between 10 lakhs and 20 lakhs.

# # LEVEL2 : DIVARIATE ANALYSIS

# # analysing the count of children and target variable

# In[66]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['CNT_CHILDREN'], hue = 'TARGET', data = df)
plt.show()


# ## INTERPRETATION : 
# * maximum clients with 0 payment difficulties have 0 children.
# * maximum clients with payment difficulties also have 0 children.

# # analysing the target variable with contract type

# In[67]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'NAME_CONTRACT_TYPE', data = df)
plt.title('count of clients according to contract type')
plt.show()


# ## INTERPRETATION :
# * maximum clients are of cash loans contract type

# # analysing the clients having car

# In[68]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'Car', data = df)
plt.title('count of clients having car')
plt.show()


# ## INTERPRETATION :
# * maximum clients have cars.

# # analysing the clients having car

# In[72]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'House', data = df)
plt.title('count of clients having House')
plt.show()


# ## INTERPRETATION :
# * Maximum clients have house.

# In[73]:


ic_corr = df['AMT_INCOME_TOTAL'].corr(df['AMT_CREDIT'])
sb.regplot(x = df['AMT_INCOME_TOTAL'], y = df['AMT_CREDIT'], label = 'R = ' +str(ic_corr), color = 'g')
plt.legend()
plt.show()


# # analysing the total income of client
# 

# In[92]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df, x = 'AMT_INCOME_TOTAL', hue = 'TARGET', bins = [100000, 200000,300000,400000,500000,600000,700000,800000])
plt.title('types of clients with their total income')
plt.show()


# ## INTERPRETATION :
# * non - defaulters have income between 1lakh and 7 lakhs, but maximum have income between 1 lakh and 2 lakhs.
# * defaulters have income between 1 lakh and 3 lakhs.
# * non- defaluters having income between 1 and 2 lakhs can also become defaulters.

# In[75]:


# calculating the measures of central tendencies
df.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].mean()


# In[76]:


df.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].median()


# In[77]:


df.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].apply(lambda x : x.mode())


# # analysing the credit amount of loan with the type of client

# In[95]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df, x = 'AMT_CREDIT', hue = 'TARGET', bins = [100000, 200000,300000,400000,500000,600000,700000,800000,900000,1000000, 1100000,1200000])
plt.title('types of clients with the credit amount of loan')
plt.show()


# ## INTERPRETATAION :
# *  the range of loan amount of non- defaulter clients is 1 lakh to 12 lakhs.
# *  the range of loan amount of defaulter clients is 1 lakh to 12 lakhs.
# 
# 

# # analysing the target variable with the price of goods

# In[79]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df, x = 'AMT_GOODS_PRICE', hue = 'TARGET', bins = 10)
plt.title('count of different clients with the good price')
plt.show()


# ## INTERPRETATAION :
# *  the amount of good price  of maximum non- defaulter clients is 5 less than 5 lakhs 
# * the amount of good price of maximum defaulter clients is also less than 5 lakhs.
# 

# # analysing the count of clients with their highest education 

# In[80]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'NAME_EDUCATION_TYPE', data = df)
plt.title('count of clients with their highest education')
plt.show()


# ## INTERPRETATIN :
# * maximum clients have secondary/secondary special education.

# # analysing count of clients and their family status

# In[81]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'NAME_FAMILY_STATUS', data = df)
plt.title('count of clients with their family status')
plt.show()


# # INTERPRETATION :
# * maximum clients are married

# # analysing target variable with days employed

# In[82]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df, x = 'DAYS_EMPLOYED', hue = 'TARGET', bins = [2500,5000,7500,10000,12500,15000,17500, 20000])
plt.title('count of different clients with the days employed')
plt.show()


# ## INTERPRETATION:
# * maximum defaulter clients applied for loan after 5000 days.
# * maximum non-defaulter clients applied for loan after 5000 days.

# # analysing clients and their occupation

# In[83]:


fig, ax = plt.subplots(figsize = (15,7))
sb.countplot(x = df['TARGET'], hue = 'OCCUPATION_TYPE', data = df)
plt.title('count of clients and their occupation')
plt.show()


# ## INTERPRETATION :
# * maximum clients are laborers.

# # MULTIVARIATE ANALYSIS

# ## filtering the following conditions in the data based on bivariate analysis.
# * count of children = 0(maximum clients have 0 children)
# * maximum clients have secondary/secondary special as highest education 
# * maximum clients are married
# * maximum clients have labourers as their occupation
# * maximum clients have submitted only one document

# In[84]:


df_1 = df[(df['CNT_CHILDREN'] == 0)&(df['NAME_EDUCATION_TYPE'] == 'Secondary / secondary special')&
         (df['NAME_FAMILY_STATUS'] == 'Married') & (df['OCCUPATION_TYPE'] == 'Laborers') & 
         (df['TOTAL_DOC_SUBMITTED'] == 1)]
df_1


# In[85]:


# filtering the clients which have car and house
df_2 = df_1[(df_1['Car'] == 'Y') & (df_1['House'] == 'Y')]
df_2


# # analysing the the total income and target 
# 

# In[89]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_2, x = 'AMT_INCOME_TOTAL', hue = 'TARGET', bins = [100000,200000,300000,400000,500000, 600000])
plt.title('count of different clients with their total income')
plt.show()


# ## INTERPRETATION :
# clients having both car and house
#  * the non - defaulter clients have income between 1 lakhs and 6 lakhs.
#  * the defaulter clients have income between 1 lakhs and 3 lakhs .
#  * the non-defaulter clients having total income between 1 lakhs and 2 lakhs can also become the defaulter.

# In[87]:


df_2.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].mean()


# In[90]:


df_2.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].median()


# In[91]:


df_2.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].apply(lambda x : x.mode())


# # filtering the clients having total income between 2 lakhs and 8 lakhs

# df_3 = df_2[(df_2['AMT_INCOME_TOTAL'] >= 200000) & (df_2['AMT_INCOME_TOTAL'] <= 800000)]
# df_3

# # analysing target and credit amount

# In[102]:


fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.histplot(data = df_2, x = 'AMT_CREDIT', hue = 'TARGET', bins = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000], ax = ax[0])
sb.boxplot(data = df_2, x ='TARGET', y ='AMT_CREDIT', ax = ax[1] )
plt.title('count of different clients with their credit amount')
plt.show()


# ## INTERPRETATION :
# * The credit amount of non - defaulter clients is between the 5 lakhs and 25 lakhs.
# * The credit amount of defaulter clients is between 5 lakhs and 15 lakhs.
# * There is a possibility that  non- defaulter clients having credit amount between 5 lakhs and 15 lakhs can become the defaulter ones.

# # analysing target and good price
# 

# In[103]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_3, x = 'AMT_GOODS_PRICE', hue = 'TARGET', bins = [500000,1000000,1500000,2000000,2500000])
plt.title('count of different clients with their good price')
plt.show()


# ## INTERPRETATION :
# * The good price  of non - defaulter clients is between the 5 lakhs and 25 lakhs.
# * The good price of defaulter clients is between 50 lakhs and 15 lakhs.
# * There is a possibility that  non- defaulter clients having good price between 5 lakhs and 15 lakhs can become the defaulter ones.

# In[104]:


# filtering the clients which do not have car and house
df_4 = df_1[(df_1['Car'] == 'N') & (df_1['House'] == 'N')]
df_4


# # analysing the clients which do not have car and house with their total income.

# In[105]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_4, x = 'AMT_INCOME_TOTAL', hue = 'TARGET',bins = [100000, 200000,300000,400000,500000,600000,700000,800000])
plt.title('count of different clients with their total income')
plt.show()


# ## INTERPRETATION :
# clients which do not have car and house :
# * non- defaulter clients have total income between 1 lakh and 5 lakhs.
# * defaulter clients have income between 1 lakh and 3 lakhs.
# * non - defaulter clients which have total income between 1 lakh and 3 lakh can also become the defaulters.

# In[106]:


df_4.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].mean()


# In[107]:


df_4.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].median()


# In[108]:


df_4.groupby(['TARGET'])[['AMT_INCOME_TOTAL']].apply(lambda x : x.mode())


# # analysing the clients which have total income between 1 lakh and 5 lakhs.

# In[ ]:


df_5 = df_4[(df_4['AMT_INCOME_TOTAL'] >= 100000) & (df_4['AMT_INCOME_TOTAL'] <= 500000 )]
df_5


# # analysing the credit amount 

# In[111]:


fig, ax = plt.subplots(1,2,figsize = (15,7))
sb.histplot(data = df_4, x = 'AMT_CREDIT', hue = 'TARGET', ax= ax[0], bins = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000])#,1200000,1400000, 1600000,1800000,2000000])
sb.boxplot(data = df_4, x ='TARGET', y ='AMT_CREDIT', ax = ax[1] )
plt.title('count of different clients with their credit amount')
plt.show()


# ## INTERPRETATION :
# * non - defaulter clients have credit amount between 2 lakhs and 20 lakhs.
# * defaulter clients have credit amount between 2 lakhs and 16 lakhs.
# * there is a possibility that non-defaulter clients having credit amount between 2 lakhs and 16 lakhs can become the defaulters.

# # analysing the good price of clients haaving total income between 1 lakh and 5 lakhs

# In[ ]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_5, x = 'AMT_GOODS_PRICE', hue = 'TARGET', bins = [500000,1000000,1500000,2000000,2500000])
plt.title('count of different clients with their good price having income between 1 lakh and 5 lakhs')
plt.show()


# ## INTERPRETATION:
# * clients having amount of good price between 5 lakhs and 25 lakhs are non-defaulters.
# * clients havinh amount of good price between 5 lakhs nad 15 lakhs are defaulters.
# * non-defaulter clients having good price between 5 lakh and 15 lakhs can become the defaulters.

# In[ ]:


# analysing the clients having either car or house


# In[112]:


df_6 = df_1[((df_1['Car'] == 'Y') & (df_1['House'] == 'N')) | ((df_1['Car'] == 'N') & (df_1['House'] == 'Y'))]
df_6


# # analysing the income of clients having either car or house

# In[113]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_6, x = 'AMT_INCOME_TOTAL', hue = 'TARGET',bins = [100000, 200000,300000,400000,500000,600000,700000,800000])
plt.title('count of different clients with their total income')
plt.show()


# ## INTERPRETATION :
#  clients which either have car or house are :
# * non- defaulters with total income between 1 lakh and 5 lakhs.
# * defaulters with total income between 1 lakh and 3 lakhs.
# non-defaulter clients having income between 1 lakh and 3 lakhs can become defaulter ones.
#     

# # analysing the clients having either house or car and total income between 1 lakh and 5 lakhs.

# In[ ]:


df_7 = df_6[(df_6['AMT_CREDIT'] >= 100000) & (df_6['AMT_CREDIT'] <=500000)]
df_7


# # analysing the type of client and loan amount

# In[116]:


fig, ax = plt.subplots(1,2,figsize = (15,7))
sb.histplot(data = df_6, x = 'AMT_CREDIT', hue = 'TARGET',ax= ax[0], bins = [100000, 150000,200000,250000,300000,350000,400000,450000,500000])
sb.boxplot(data = df_6, x ='TARGET', y ='AMT_CREDIT', ax = ax[1] )
plt.title('count of different clients with their credit amount')
plt.show()


# ## INTERPRETATION :
# * clients wether they are defaulter and non- defaulter have credit amount between 1 lakh and 5 lakh.

# # analysing the good price

# In[ ]:


fig, ax = plt.subplots(figsize = (15,7))
sb.histplot(data = df_7, x = 'AMT_GOODS_PRICE', hue = 'TARGET', bins = [50000,100000,150000,200000,250000, 300000,350000,400000])
plt.title('count of different clients with their good price having income between 1 lakh and 5 lakhs')
plt.show()


# ## INTERPRETATION :
# * non-defaulter clients have good price amount between 50 thousand and 4 lakh.
# * defaulters have amount of good price between 1 lakh and 4 lakhs.
# * maximum defaulters have good price amount between 2 lakhs and 2.5 lakhs, therefore non-defaulter clients having good price between 2 lakhs and 2.5 lakhs can become the defaulters.
