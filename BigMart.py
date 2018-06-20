
pathname="G:\\BigMart\\" #where submission file is to be saved
path='G:\\BigMart' #Where Files are located
# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.mlab as mlab

############# load dataset##########################################################################
def upload_csv(path,filename):
    fileaddress=path+'/'+filename
    return pd.read_csv(fileaddress,na_values={'Outlet_Size': []})
    


train=upload_csv(path,'train.csv')
test=upload_csv(path,'test.csv')

train['source']='Train' #Create a flag for Train and Test Data set
test['source']='Test'

fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set

def get_var_category(series):
    unique_count = series.nunique(dropna=False)
    total_count = len(series)
    if pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    elif pd.api.types.is_datetime64_dtype(series):
        return 'Date'
    elif unique_count==total_count:
        return 'Text (Unique)'
    else:
        return 'Categorical'

def print_categories(df):
    for column_name in df.columns:
        print(column_name, ": ", get_var_category(df[column_name]))

def missing_val(dataset):
    x=dataset[dataset['source']=='Train'].isnull().sum()
    temp1=pd.DataFrame({'Features':x.index,'Train':x.values})
    x=dataset[dataset['source']=='Test'].isnull().sum()
    temp2=pd.DataFrame({'Features':x.index,'Test':x.values})
    print(pd.merge(temp1,temp2,on=['Features'],how='left'))

def description(df):
    print(df.describe())

def eda_category(dataset):
    for col in dataset.select_dtypes(include=['object']).columns:
            print(dataset[col].value_counts(sort=True ,dropna=False))


#################################################################################################### 
print_categories(train)
description(train)
missing_val(fullData)
description(train)
eda_category(fullData.drop(['Item_Identifier','Outlet_Identifier'],axis=1))

#Treating Missing Values
ID_col='Item_Identifier'
target_col='Item_Outlet_Sales'

miss_bool = fullData['Item_Weight'].isnull() 
item_avg_weight = fullData.pivot_table(values='Item_Weight', index='Item_Identifier')
fullData.loc[miss_bool,'Item_Weight'] = fullData.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])

#Determing the mode for each  
from scipy.stats import mode

def dataWithoutNull(df):
    d=df.dropna()
    return mode(d)[0]

#fullData['Outlet_Size']=fullData['Outlet_Size'].astype(str)
outlet_size_mode = fullData.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:dataWithoutNull(x)))
print('Mode for each Outlet_Type:')
outlet_size_mode=outlet_size_mode.transpose()
#Get a boolean variable specifying missing Outletsize values
miss_bool = fullData['Outlet_Size'].isnull()
fullData.loc[miss_bool,'Outlet_Size'] = fullData.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode.loc[x])

#Determine average visibility of a product
visibility_avg = fullData.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (fullData['Item_Visibility'] == 0)
fullData.loc[miss_bool,'Item_Visibility'] = fullData.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])

#Determine another variable with means ratio
fullData['Item_Visibility_MeanRatio'] = fullData.apply(lambda x: x.loc['Item_Visibility']/visibility_avg.loc[x.loc['Item_Identifier']], axis=1)
print(fullData['Item_Visibility_MeanRatio'].describe())

data=fullData
#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()



data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})


#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data[data['source']=="Train"]
test = data[data['source']=="Test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source','SKU'],axis=1,inplace=True)
train.drop(['source','SKU'],axis=1,inplace=True)

#Export files as modified versions:
trainadd=pathname+"train_modified.csv"
testadd=pathname+"test_modified.csv"
train.to_csv(trainadd,index=False)
test.to_csv(testadd,index=False)

#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
sub1=pathname+"alg0.csv"
base1.to_csv(sub1,index=False)

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    filename=pathname+filename
    submission.to_csv(filename, index=False)



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn') # pretty matplotlib plots

plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


import statsmodels.api as sm
X = sm.add_constant(train[predictors])
model = sm.OLS(train[target], X).fit()
model.summary()
train.drop(['residual'],axis=1,inplace=True)
corr=np.corrcoef(train[predictors].select_dtypes(exclude=[object]),rowvar=0)
W,V=np.linalg.eig(corr)
#linear Regression diagnoistic
train['residual']=train[target]-alg1.predict(train[predictors])
plt.plot(alg1.predict(train[predictors]),train['residual'])

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')


from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')



from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')

