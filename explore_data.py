import pandas as pd


#loadthedata
df=pd.read_csv('Churn_Modelling.csv')

#look at the first 5 rows
print(df.head())

#check basic info-column types,missing values
print(df.info())

#check for any missing values
print(df.isnull().sum())

#drop column we don't need for analysis
df_clean=df.drop(['RowNumber','CustomerId','Surname'],axis=1)

#cofirm it worked
print(df_clean.head())
print(df_clean.shape)

#convert gender to numbers: female=0, male=1
df_clean['Gender']=df_clean['Gender'].map({'Female':0,'Male':1})

#convert Geography into seperate yes/no columns(one-hot encoding)
df_clean=pd.get_dummies(df_clean,columns=['Geography'],drop_first=True)

#confirm if worked
print(df_clean.head())
print(df_clean.columns.tolist())

#create age groups
def age_group(age):
    if age<=30:
        return 'Young'
    elif age<=45:
        return 'Middle'
    elif age<=60:
        return 'Senior'
    else:
        return 'Elder'

df_clean['AgeGroup']=df_clean['Age'].apply(age_group)

#simple CLV estimate: Balance + EstimatedSalary*Tenure*0.1
df_clean['CLV']=df_clean['Balance']+(df_clean['EstimatedSalary']*df_clean['Tenure']*0.1)

#confirm
print(df_clean[['Age','AgeGroup','Balance','Tenure','EstimatedSalary','CLV']].head())


# Split data for training, and build the churn prediction model(this is the actual machine learning step)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#features(inputs)- evrything except what we are predicting and non-usefull text columns
X=df_clean.drop(['Exited','AgeGroup'], axis=1)

#target(what we are predicting)
y=df_clean['Exited']

from sklearn.preprocessing import StandardScaler

#scale the features so they're all on a similar range
scaler =StandardScaler()
X_scaled=scaler.fit_transform(X)

#split into training data (80%) and testing data (20%)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#create and train model 
model=LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train,y_train)

#Make predictions on test data 
y_pred=model.predict(X_test)


#check how well it performed
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


#get churn probability for all 10,000 customers(not just test set)
df_clean['ChurnProbability']=model.predict_proba(X_scaled)[:,1]

#flag high risk customers (probability >0.5)
df_clean['ChurnRishkFlag']=(df_clean['ChurnProbability']>0.5).astype(int)

#bring back customerid so power bi can identify each customer
df_clean['CustomerId']=df['CustomerId']

#check what the final columns look like 
print(df_clean.head())
print(df_clean.columns.tolist())

#export the final file for Power BI
df_clean.to_csv('churn_scored_data.csv',index=False)
print("Exported Successfully!")

import pandas as pd

#Get feature importance from the trained model
feature_importance=pd.DataFrame({ 'Feature': X.columns,
                                 'Coefficient': model.coef_[0]})

#sort by absolute impact (strongest predictors first positive or negative)
feature_importance['AbsCoefficient']=feature_importance['Coefficient'].abs()
feature_importance=feature_importance.sort_values('AbsCoefficient',ascending=False)

print(feature_importance)

#Export this separately for Power BI
feature_importance.to_csv('Feature_importance.csv',index=False)