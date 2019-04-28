
# coding: utf-8

# In[2]:


import os 
os.chdir("C:/Users/Dell/Documents/Bikesharing")


# In[14]:


os.getcwd()


# In[15]:


import os 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


data=pd.read_csv("day.csv",sep=',')


# In[17]:


data.head()


# In[18]:


data.tail()


# In[19]:


data.rename(columns={'yr':'year','mnth':'month','cnt':'count','dteday':'dateday'},inplace=True) 


# In[20]:


data.shape


# In[21]:


data.isnull().sum()


# In[22]:


data.columns


# In[23]:


continous = ['temp' , 'atemp' , 'hum' , 'windspeed' , 'casual' , 'registered' , 'count']


# In[24]:


continous


# In[25]:


data[continous].describe()


# In[26]:


for i , name in enumerate(continous): 
        plt.figure(figsize=(15,20)) 
        plt.subplots_adjust(hspace=0.7, wspace=0.7) 
        b = ''    
        b += name 
        plt.figure()    
        plt.title('Box plot of {}'.format(b)) 
        sns.boxplot(y = data[b])    
        plt.show() 


# In[55]:


data[ (data['instant']>46)& (data['instant']<53)]
     


# In[27]:


data['hum'].replace(0.187917, 0.507463, inplace=True)
data['windspeed'].replace(0.507463, 0.187917, inplace=True)


# In[28]:


mean_humidity=data[(data['instant'] >  62) & (data['instant'] < 75)] 


# In[29]:


mean_humidity['hum'].describe()


# In[30]:


data['hum'].replace(0,60250, inplace=True)


# In[31]:


plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.7, wspace=0.7)
plt.subplot(221)
sns.boxplot(y = data['hum'],color = 'brown')
plt.ylabel('humidity')
plt.title('plot of humidity')

plt.subplot(222)
sns.boxplot(y = data['windspeed'], color = 'green')
plt.ylabel('windspeed')
plt.title('plot of windspeed')
plt.show()




# In[33]:


plt.figure(figsize=(14, 15))
plt.subplots_adjust(hspace=0.7, wspace=0.7)
plt.subplot(441)
sns.distplot(data['temp'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('Temperature Distribution')

plt.subplot(442)
sns.distplot(data['windspeed'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('windspeed Distribution')

plt.subplot(443)
sns.distplot(data['hum'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('Humidity Distribution')
plt.show()


# In[34]:


plt.figure(figsize=(14, 15))
plt.subplots_adjust(hspace=0.7, wspace=0.7)

plt.subplot(441)
sns.distplot(data['count'], hist=False, rug=True,color='green') 
plt.ylabel('density')
plt.title('count')
plt.show()


# In[35]:


data.columns


# In[36]:


data['season'].value_counts()


# In[38]:


plt.figure(figsize=(26,26))
plt.subplot(441) 
plt.subplots_adjust(hspace=0.7, wspace=0.7) 
#plt.ylabel('Density') 
plt.title('workingdays Distribution') 
sns.countplot(data['workingday'],color='brown',order = data['workingday'].value_counts().index) 

#plt.xticks(rotation=45) 

plt.subplot(442) 
#plt.ylabel('Density') 
plt.title('weathersit Distribution')
sns.countplot(data['weathersit'],color='green',order=data['weathersit'].value_counts().index) 

plt.subplot(443) 
#plt.ylabel('Density') 
plt.title('holiday Distribution') 
sns.countplot(data['holiday'],color='orange',order=data['holiday'].value_counts().index) 
plt.show()


# In[27]:


#casual, registered visualization


# In[39]:


data.columns


# In[40]:


data['dateday']=data['dateday'].astype('datetime64')


# In[41]:


data_30 = data[:24]


# In[42]:


plt.figure(figsize=(16,6))
plt.plot(data_30['dateday'],data_30['count'])
plt.xlabel('date')
plt.ylabel('count')
plt.title('Date(24days) vs count')
plt.show()


# In[43]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(16,6)) 
plt.plot(data['dateday'],data['casual']) 
plt.plot(data['dateday'],data['registered'],color = 'orange') 
plt.xlabel('date') 
plt.ylabel('Ride_count')
plt.title('Date vs Bicycle ride count') 
plt.legend() 
plt.show()


# In[44]:


plt.figure(figsize=(16,6)) 
plt.plot(data['dateday'],data['registered'],color = 'orange')
plt.xlabel('date') 
plt.ylabel('Registered')
plt.title('Date vs Registered') 
plt.show()


# In[45]:


plt.figure(figsize=(16,6)) 
plt.plot(data['dateday'],data['count'],color = 'green') 
plt.xlabel('date')
plt.ylabel('Total_count') 
plt.title('Date vs Total_count') 
plt.show()


# In[46]:


month_casual = data.groupby(['year','month'])['casual','registered'].mean() 
month_casual1 = data.groupby(['weekday'])['casual','registered','count'].mean() 


# In[47]:


month_casual1


# In[48]:


month_casual = month_casual.reset_index() 


# In[49]:


x = month_casual.reset_index()
x.drop(['year','month','index'],axis=1,inplace=True) 


# In[50]:


x.plot(title='Avg use by month') 
plt.xlabel('Month')
plt.ylabel('Avgerage count') 
plt.show()


# In[51]:


month_casual1 = data.groupby(['weekday'])['casual','registered'].mean() 


# In[52]:


month_casual1.plot(title='Average use by day') 
plt.show()


# In[53]:


data['week_day'] = (data['weekday'] > 5) & (data['weekday'] == 0)
data['week_day'] =  (data['weekday'] >= 1) & (data['weekday'] <= 5) 


# In[54]:


x =[] 
for i in data['week_day']:  
 if  i is False:       
    x.append(1)   
 else:   
    x.append(0) 


# In[55]:


data['week_end'] = x 


# In[56]:


data_weekbday = data.pivot_table('count',aggfunc='sum',index=['year','month'],columns='week_day') 


# In[57]:


data_weekday  = data_weekbday.reset_index() 
data_weekday.drop(['year','month'],axis=1,inplace=True) 


# In[59]:


data_weekday.plot() 
plt.xlabel('Month') 
plt.ylabel('Total Count') 
plt.title('(Weekday) Month vs Total Count') 
plt.show()


# In[60]:


casual_pattern_weekday = data.pivot_table('casual',aggfunc='sum',index=['year','month'],columns='week_day') 


# In[61]:


casual_pattern_weekday = casual_pattern_weekday.reset_index() 
casual_pattern_weekday.drop(['year','month'],axis=1,inplace=True) 
casual_pattern_weekday.plot() 
plt.xlabel('Month') 
plt.ylabel('casual_count')
plt.title('Month vs Casual')
plt.show()


# In[62]:


data.columns


# In[63]:


holidays = data.pivot_table('count',aggfunc='mean',index=['month'],columns='holiday') 


# In[64]:


holidays.dropna(inplace = True)


# In[65]:


holidays = holidays.sort_values(by=1, ascending =False)


# In[66]:


holidays = holidays.reset_index()


# In[67]:


holidays


# In[68]:


sns.barplot(holidays.month,holidays[1],color="green",order=holidays.month) 
plt.xlabel('Holiday Month') 
plt.ylabel('Count') 
plt.title('Holiday Month vs Count') 
plt.show()


# In[69]:


sns.violinplot(data['weathersit'],data['count'])
plt.title('Violin plot of weathersit vs count') 
plt.show()


# In[70]:


sns.violinplot(data['season'],data['count']) 
plt.title('Violin plot of season vs count') 
plt.show()


# In[71]:


sns.lmplot('temp','count',data=data) 
plt.xlabel('Temperature') 
plt.ylabel('Count') 
plt.title('Temperature vs Total Count') 
plt.show()


# In[72]:


data.groupby('season')['temp'].mean() 


# In[73]:


t = sns.lmplot(x="temp", y="count", hue="season", col="season",             
                  data=data, aspect=.4, x_jitter=.1) 


# In[74]:


sns.lmplot('hum','count',data=data) 
plt.xlabel('Humidity') 
plt.ylabel('Count') 
plt.title('Humidity vs Count')
plt.show()
    


# In[75]:


data.groupby('season')['hum'].median() 


# In[76]:


t = sns.lmplot(x='hum', y="count", hue="season", col="season",              
               data=data, aspect=.4, x_jitter=.1) 


# In[77]:


data.columns 


# In[78]:


data['week_day'].value_counts() 


# In[79]:


names =  ['season', 'year', 'month', 'holiday', 'weekday',       
          'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed','count'] 


# In[80]:


co = data[names].corr() 
    
correlation = co
#plt.figure(figsize=(10,10)) 
plt.figure(figsize = (20,20)) 
g = sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix',xticklabels=True,yticklabels=True)
g.set_yticklabels(g.get_yticklabels(), rotation =0) 
g.set_xticklabels(g.get_yticklabels(), rotation =90)
plt.title('Correlation between different fearures')
plt.show()


# In[110]:


#model


# In[81]:


data.columns


# In[82]:


x=pd.DatetimeIndex(data['dateday']) 


# In[83]:


x = x.day 
data['day'] = x 


# In[84]:


columns = ['dateday','day','season', 'year', 'month', 'holiday', 'weekday',       
           'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'count'] 


# In[85]:


data_train = data[columns] 


# In[86]:


weather_sit = pd.get_dummies(data_train['weathersit'],prefix='weathersit') 
seasons_dummy = pd.get_dummies(data_train['season'],prefix='season') 
weekday_dummy = pd.get_dummies(data_train['weekday'],prefix='weekday') 


# In[87]:


weather_sit.drop(['weathersit_3'],axis=1,inplace=True) 
seasons_dummy.drop(['season_4'],axis=1,inplace=True) 
weekday_dummy.drop(['weekday_6'],axis=1,inplace=True)


# In[88]:


def nomalizar(x):    
       norm  = (x-min(x))/(max(x)-min(x))    
       return norm 


# In[89]:


c = nomalizar(data['month']) 


# In[90]:


data_train['month_norm'] = c 


# In[91]:


data_train1 = pd.concat([data_train,weather_sit,seasons_dummy,weekday_dummy],axis=1) 


# In[92]:


data_train1.columns 


# In[93]:


names_columns =['temp', 'hum', 'windspeed','workingday','holiday',        
                'weathersit_1', 'weathersit_2', 'season_1', 'season_2',       
                'season_3', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',       
                'weekday_4', 'weekday_5']


# In[94]:


test = data_train1.iloc[710:,:] 


# In[95]:


train = data_train1.iloc[:710,:] 


# In[96]:


from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV 


# In[97]:


Reg = LinearRegression() 


# In[98]:


Reg.fit(train[names_columns],train['count']) 


# In[99]:


y_pred = Reg.predict(test[names_columns]) 


# In[101]:


from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 


# In[102]:


np.sqrt(mean_squared_error(test['count'],y_pred)) 


# In[103]:


r2_score(test['count'],y_pred) 


# In[104]:


importance = Reg.coef_ 

plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.7) 
plt.subplot2grid((3,2),(0,1)) 
importance = pd.DataFrame({'importance':importance,'feature':names_columns}) 
importance = importance.sort_values(by ='importance',ascending=False) 
sns.barplot(importance['importance'],importance['feature']) 
#plt.xticks(rotation =90)
plt.title('LR feature importance') 
plt.show()


# In[105]:


plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test')
plt.plot(test['day'],y_pred,label='predicted')
plt.legend()
plt.xlabel('Days')
plt.ylabel('count')
plt.title('LR Actual vs Predicted')
plt.show()


# In[106]:


data.columns


# In[107]:


columns


# In[108]:


columns =['season', 'holiday', 'weekday',       'workingday', 'weathersit', 'temp', 'hum', 'windspeed'] 


# In[109]:


test = data.iloc[710:,:]
train = data.iloc[:710, :]


# In[110]:


from sklearn.svm import LinearSVR



# In[111]:


Reg = LinearSVR()


# In[112]:


param_grid2 = {'C':[0.05,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,16,17]           
                            }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3) 
best_model= grid_search.fit(train[columns],train['count']) 


# In[113]:


best_model.best_params_ 


# In[114]:


Reg = LinearSVR(C=17) 


# In[115]:


Reg.fit(train[columns],train['count'])
y_pred = Reg.predict(test[columns]) 


# In[116]:


print(np.sqrt(mean_squared_error(test['count'],y_pred)))
print(r2_score(test['count'],y_pred)) 


# In[117]:


importance= Reg.coef_ 
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False) 
importance 


# In[118]:


sns.barplot(importance['importance'],importance['feature'],color='blue') 
plt.title('SVM feature importance') 
plt.show()


# In[119]:


plt.figure(figsize=(10,3)) 
plt.plot(test['day'],test['count'],label='test',color='red')
plt.plot(test['day'],y_pred,label='predicted') 
plt.legend()
plt.xlabel('Days') 
plt.ylabel('Count') 
plt.title('(SVM) Actual vs Predicted')
plt.show()


# In[104]:





# In[120]:


from sklearn.tree import DecisionTreeRegressor 


# In[121]:


Reg = DecisionTreeRegressor()


# In[122]:


param_grid2 = {'max_depth':[None,3,5,6,8,9,10,12,15,17,18,20],             
               # "min_samples_split": [2,3,4,5,6,7,8,10,15,20]                 
               #"min_samples_leaf": [1,2,3,4,5,10,30]              
              }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3) 
best_model= grid_search.fit(train[columns],train['count']) 


# In[123]:


best_model.best_params_


# In[124]:


best_model.best_params_


# In[116]:


best_model.best_params_


# In[125]:


Reg = DecisionTreeRegressor(max_depth=6,min_samples_split=4,min_samples_leaf=1) 


# In[126]:


Reg.fit(train[columns],train['count'])
y_pred = Reg.predict(test[columns]) 


# In[127]:


print(np.sqrt(mean_squared_error(test['count'],y_pred))) 
print(r2_score(test['count'],y_pred)) 


# In[120]:





# In[128]:


sns.barplot(importance['importance'],importance['feature'],color='grey') 
plt.title('Decision Tree Feature Imp') 
plt.show()


# In[129]:


plt.figure(figsize=(10,3)) 
plt.plot(test['day'],test['count'],label='test',color='black')
plt.plot(test['day'],y_pred,label='predicted',color='red') 
plt.legend() 
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(Decision tree) Actual vs Predicted') 
plt.show()


# In[130]:


from sklearn.ensemble import AdaBoostRegressor 


# In[131]:


Reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=6)) 


# In[132]:


param_grid2 = {"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500]
               
              }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3) 
best_model= grid_search.fit(train[columns],train['count']) 


# In[133]:


best_model.best_params_ 


# In[134]:


Reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=6),n_estimators=150) 


# In[135]:


Reg.fit(train[columns],train['count']) 
y_pred = Reg.predict(test[columns]) 


# In[136]:


print(np.sqrt(mean_squared_error(test['count'],y_pred))) 
print(r2_score(test['count'],y_pred)) 


# In[137]:


plt.figure(figsize=(10,3)) 
plt.plot(test['day'],test['count'],label='test')
plt.plot(test['day'],y_pred,label='predicted',color='green') 
plt.legend()
plt.xlabel('Days') 
plt.ylabel('Count')
plt.title('(Adaboost) Actual vs Predicted') 
plt.show()


# In[138]:


Reg.feature_importances_


# In[139]:


importance= Reg.feature_importances_ 
importance = pd.DataFrame({'importance':importance,'feature':columns}) 
importance = importance.sort_values(by ='importance',ascending=False) 
importance 


# In[140]:


sns.barplot(importance['importance'],importance['feature'],color='purple') 
plt.title('Adboost with Decision Tree Feature Imp') 
plt.show()


# In[141]:


data.columns 


# In[142]:


columns =['season', 'holiday', 'weekday',       'workingday', 'weathersit', 'temp', 'hum', 'windspeed'] 


# In[143]:


test = data.iloc[:710,:]


# In[144]:


train = data.iloc[:710,:] 


# In[145]:


from sklearn.ensemble import RandomForestRegressor


# In[146]:


Reg = RandomForestRegressor(n_jobs=-1) 


# In[147]:


param_grid2 = {#"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500],            
#"max_depth": [None,3,5,6,8,9,10,12,15,17,18,20],              
   
#"min_samples_split": [2,3,4,5,6,7,8,10,15,20],             
     "min_samples_leaf": [1,2,3,4,5,10,30],             
    # "max_leaf_nodes": [None,5,10,20,30, 40],            
    # "max_features": ['auto',0.5,'log2']              
            } 
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3) 
best_model= grid_search.fit(train[columns],train['count']) 


# In[148]:


best_model.best_params_ 


# In[149]:


best_model.best_params_ 


# In[150]:


Reg = RandomForestRegressor(n_jobs=-1,n_estimators=260,max_depth=10,min_impurity_decrease=4,min_samples_leaf= 1, 
                            bootstrap=True) 

   


# In[151]:


Reg.fit(train[columns],train['count']) 


# In[152]:


y_pred= Reg.predict(test[columns]) 


# In[153]:


r2_score(test['count'],y_pred) 


# In[154]:


np.sqrt(mean_squared_error(test['count'],y_pred)) 


# In[155]:


importance= Reg.feature_importances_


# In[156]:


importance = pd.DataFrame({'importance':importance,'feature':columns}) 


# In[157]:


importance = importance.sort_values(by ='importance',ascending=False) 


# In[158]:


sns.barplot(importance['importance'],importance['feature']) 
plt.title('RF feature imp') 
plt.show()


# In[202]:


importance


# In[205]:


plt.figure(figsize=(10,3)) 
plt.plot(test['day'],test['count'],label='test',color='blue')
plt.plot(test['day'],y_pred,label='predicted',color='red') 
plt.legend() 
plt.xlabel('Days') 
plt.ylabel('Count')
plt.title('(RF) Actual vs Predicted') 
plt.show()


# In[159]:


from sklearn.ensemble import ExtraTreesRegressor 


# In[160]:


Reg = ExtraTreesRegressor(n_jobs=-1,n_estimators=30,max_depth=12,min_impurity_decrease=3,min_samples_leaf=1) 


# In[161]:


param_grid2 = {#"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500],            
#"max_depth": [None,3,5,6,8,9,10,12,15,17,18,20],             
#"min_samples_split": [2,3,4,5,6,7,8,10,15,20],             
#"min_samples_leaf": [1,2,3,4,5,10,30],             
    "max_leaf_nodes": [None,5,10,20,30, 40],             
    #"max_features": ['auto',0.5,'log2']             
      } 
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count']) 


# In[162]:


best_model.best_params_ 


# In[163]:


best_model.best_params_


# In[164]:


Reg = ExtraTreesRegressor(n_jobs=-1,n_estimators=50,max_depth=12,min_impurity_decrease=3,min_samples_leaf=1,                         
                          bootstrap=True)


# In[165]:



Reg.fit(train[columns],train['count']) 
y_pred= Reg.predict(test[columns]) 


# In[166]:


r2_score(test['count'],y_pred) 


# In[218]:


np.sqrt(mean_squared_error(test['count'],y_pred)) 


# In[219]:


importance= Reg.feature_importances_ 
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
importance 


# In[221]:


importance = importance.sort_values(by ='importance',ascending=False)
sns.barplot(importance['importance'],importance['feature'],color='orange')
plt.title('ETR feature importance') 
plt.show()


# In[225]:


plt.figure(figsize=(15,4)) 
plt.plot(test['day'],test['count'],label='test',color='red') 
plt.plot(test['day'],y_pred,label='predicted',color='brown') 
plt.legend() 
plt.xlabel('Days') 
plt.ylabel('Count') 
plt.title('(Extre tree) Actual vs Predicted') 
plt.show()


# In[231]:


importance= Reg.feature_importances_ 
importance = pd.DataFrame({'importance':importance,'feature':columns}) 
importance = importance.sort_values(by ='importance',ascending=False) 
sns.barplot(importance['importance'],importance['feature'],color='brown')
plt.show()


# In[167]:


plt.figure(figsize=(10,3)) 
plt.plot(test['day'],test['count'],label='test') 
plt.plot(test['day'],y_pred,label='predicted')
plt.legend()
plt.xlabel('Days') 
plt.ylabel('Count') 
plt.show()


# In[3]:


x='cvbv..hhhhh'


# In[4]:


x.split('.')


# In[6]:


for i,j in enumerate([10,20,30,14,50]):
    print(i,j)

