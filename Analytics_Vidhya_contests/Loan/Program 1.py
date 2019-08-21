##Loan Problem
#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

#Data Preprocessing 
df_train = pd.read_csv('train_ctrUa4K.csv')
df_test = pd.read_csv('test_lAUu6dG.csv')
loan_status = df_train.Loan_Status
data = pd.concat([df_train.drop(['Loan_Status'],axis=1), df_test])
data['LoanAmount']== data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
data['Loan_Amount_Term']== data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.median(),inplace=True)
data['Credit_History']== data.Credit_History.fillna(data.Credit_History.median(),inplace=True)
data = pd.get_dummies(data,columns=['Gender','Married','Self_Employed','Education'], drop_first=True)
data = data[['ApplicantIncome',
       'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Gender_Male','Married_Yes', 'Self_Employed_Yes',
       'Education_Not Graduate']]
data.sort_index()
#splitting of the datasets
data_train = data.iloc[:614]
data_test = data.iloc[614:]
X = data_train.values
selector = VarianceThreshold()
selector.fit_transform(X)
test = data_test.values
y = loan_status.values
#features Scaling
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
test = sc_x.transform(test) 
#Grid Search Crossing Validation for logestic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":[0.01,1.0,1.2,1.3], "solver":["newton-cg","lbfgs","sag"],"class_weight":["balanced"],
      "multi_class":['ovr', 'multinomial', 'auto'],"random_state":[10,20,30,40,50],"warm_start":[True]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X,y)
print("tuned hpyerparameters logestic regression :(best parameters) ",logreg_cv.best_params_)
print("logestic regression accuracy :",logreg_cv.best_score_)
#Fitting the Logestic Regression
classifier=LogisticRegression(C=0.01,solver='newton-cg',class_weight="balanced",random_state=10,multi_class="ovr",warm_start=True)
classifier.fit(X,y) 
#predection 
y_pred = classifier.predict(test)
df_test['Loan_Status']= y_pred
#XGBoosting
from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(X,y)
#kfold crossvalidation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier_xgb,X=X, y=y,cv=10)
print("Acurracy :{} and Standard Deviation :{}".format(accuracy.mean(),accuracy.std()))
#Preparing submission file 
submission = pd.DataFrame({})
submission['Loan_ID']=df_test['Loan_ID']
submission['Loan_Status']=y_pred 
submission.to_csv("logistic.csv", index=False)

