#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jetfire
"""
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, learning_curve,ShuffleSplit,cross_val_predict,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.learning_curve import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from keras.models import Sequential 
from keras.layers import Dense 
import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import seaborn as sns

#data Preprocessing 
#setting the grid Theme for learning curve and valdidation curve
sns.set_style(style= 'darkgrid')
Path1 = "census.csv"
#importing the file using Pandas Library
df = pd.read_csv(Path1, header=None)
header=["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","gender","capital-gain","capital-loss",
        "hours-per-week","native-country","Above_50k"]
df.columns=header
df.drop(columns=['fnlwgt','education-num'], axis = 1, inplace = True)
df['Above_50k'] = df['Above_50k'].isin([' >50K'])
df["workclass"]=df["workclass"].replace(' ?', np.NaN)
df["education"]=df["education"].replace(' ?', np.NaN)
df["occupation"]=df["occupation"].replace(' ?', np.NaN)
df["native-country"]=df["native-country"].replace(' ?', np.NAN)
df.dropna(inplace=True)
print("DataSet Loaded ")
#data preprocessing for dataset1
X = df.iloc[:,0:11].values #independent Variable
y = df.iloc[:,-1].values #dependent Variable
#Handling Null values
'''imputer = Imputer(missing_values='Nan', strategy="most_frequent", axis=0)
imputer = imputer.fit(X[:,1])
imputer = imputer.fit(X[:,2])
imputer = imputer.fit(X[:,4])
'''
labelencode_X1 = LabelEncoder()
X[:,1] = labelencode_X1.fit_transform(X[:,1])
labelencode_X2 = LabelEncoder()
X[:,2] = labelencode_X2.fit_transform(X[:,2])
labelencode_X3 = LabelEncoder()
X[:,3] = labelencode_X3.fit_transform(X[:,3])
labelencode_X4 = LabelEncoder()
X[:,4] = labelencode_X4.fit_transform(X[:,4])
labelencode_X5 = LabelEncoder()
X[:,5] = labelencode_X5.fit_transform(X[:,5])
labelencode_X6 = LabelEncoder()
X[:,6] = labelencode_X6.fit_transform(X[:,6])
labelencode_X7 = LabelEncoder()
X[:,7] = labelencode_X7.fit_transform(X[:,7])
labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)
#y = labelencode_X11.fit_transform(y1)
onehotencode = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7])
X = onehotencode.fit_transform(X).toarray()

#data split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 33)
sc = StandardScaler()
X_trainA = sc.fit_transform(X_train)
X_testA = sc.transform(X_test)
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=33)
#Modeling curves
#validation curve
def validation_curve_model(X, Y, model, param_name, parameters, cv,x_label,fig_name):

    train_scores, test_scores = validation_curve(model, X, Y, param_name=param_name, param_range=parameters,cv=cv,n_jobs=psutil.cpu_count(),scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation curve")
    plt.fill_between(parameters, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(parameters, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(parameters, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(parameters, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.ylim([0.4, 1.1])
    '''if ylim is not None:
        plt.ylim(*ylim)'''

    plt.ylabel('Score')
    plt.xlabel(x_label)
    plt.xticks(parameters)
    plt.legend(loc="best")
    plt.savefig(fig_name)
    
    return plt

#learning curve
def Learning_curve_model(X, Y, model, cv, train_sizes,fig_name):

    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")


    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")                
    plt.legend(loc="best")
    plt.savefig(fig_name)
    return plt

#prediction model
def predict_model(X, Y, model, Xtest, cv):
   model.fit(X, Y)
   Y_pred  = model.predict(Xtest)
   score   = cross_val_score(model, X, Y_pred, cv=cv)
   return score 


#Modeling functions
def decessionTree(option):
    print("DT Tree : {}".format(option))
    t_start = time.clock()
    clf = DecisionTreeClassifier(max_depth=5,min_samples_leaf=5)
    clf.fit(X_train,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf.predict(X_test)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
    '''print('Predicted score of Decision Tree classifier on Dataset: {:.2f}'
     .format(predict_model(X_train, y_train, clf, X_test)))'''
    param_range = np.arange(1, 41, 2)
    validation_curve_model(X, y, DecisionTreeClassifier(class_weight='balanced'),
                           param_name="max_depth",parameters=param_range, cv=4, x_label="max_depth",fig_name='DSSE.jpg')
    plt.show()
    train_size=np.linspace(.1, 1.0, 15)
    Learning_curve_model(X, y, DecisionTreeClassifier(class_weight='balanced'), cv=4, train_sizes=train_size,fig_name='DTLearning.jpg' )
    plt.show() 
    #scores = clf.score(X_train, y_train)
    # print("\n%s: %.2f%%" % (clf.metrics_names[1], scores[1]*100))
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy:{}".format(Pred))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('test.png')
    Image(filename='test.png')
    menu()
    
    
    
def Svm_fun(option):
    print("SVM TRee")
    t_start = time.clock()
    clf = SVC(kernel = 'rbf', random_state = 33)
    clf.fit(X_train,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf.predict(X_test)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of SVM Gaussian kernel classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of SVM Gaussin kernel classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy SVM Gaussian kernel classifier:{}".format(Pred))
    t_start = time.clock()
    clf = SVC(kernel = 'poly', random_state = 33)
    clf.fit(X_trainA,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf.predict(X_testA)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of SVM linear kernel classifier on training set: {:.2f}'
          .format(clf.score(X_trainA, y_train)))
    print('Accuracy of SVM linear kernel classifier on test set: {:.2f}'
     .format(clf.score(X_testA, y_test)))    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy SVM linear kernel classifier:{}".format(Pred))
    param_range = np.linspace(0.1, 1.0, 10)
    validation_curve_model(X, y, model=SVC(kernel='linear'), param_name="C", parameters=param_range, cv=4,x_label="itrations",fig_name='SVMVpoly.jpg')
    validation_curve_model(X, y, model=SVC(kernel='rbf'), param_name="C", parameters=param_range, cv=4,x_label="itrations",fig_name='SVMVrbf.jpg')
    plt.show()
    param_range = np.linspace(0.1, 1.0, 10)
    Learning_curve_model(X, y, model=SVC(kernel='poly'), cv=4, train_sizes=param_range, fig_name='SVMLpoly.jpg')
    Learning_curve_model(X, y, model=SVC(kernel='rbf'), cv=4, train_sizes=param_range,fig_name='SVMLrbf.jpg')
    plt.show() 
    menu()
        

def ANN_fun(option):
    print("ANN")
    t_start = time.clock()
    classifier = Sequential()
    classifier.add(Dense(30, activation='relu', input_dim=61))
    classifier.add(Dense(11, activation='relu'))
    classifier.add(Dense(11, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=classifier.fit(X_trainA, y_train, epochs=100,validation_split=0.33 ,batch_size=10, verbose=1)
    t_end = time.clock()
    p_start = time.clock()
    scores = classifier.evaluate(X_trainA, y_train)
    print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
    y_pred = classifier.predict(X_testA)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy:{}".format(Pred))
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('ANNAcc.jpg')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('epoch_Acc.jpg')
    plt.show()
    menu()


def ADAboost_fun(option):
    print("ADABoost : {}".format(option))
    t_start = time.clock()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_leaf=5),n_estimators=50,learning_rate=1,random_state=33)
    clf.fit(X_train,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf.predict(X_test)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of ADABoot classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
    print('Accuracy of ADAboost Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
    param_range = np.arange(1, 41, 2)
    validation_curve_model(X, y, AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)),
                           param_name="n_estimators",parameters=param_range, cv=4, x_label="Estimators",fig_name='ADA.jpg')
    plt.show()
    train_size=np.linspace(.1, 1.0, 15)
    Learning_curve_model(X, y, DecisionTreeClassifier(class_weight='balanced'), cv=4, train_sizes=train_size,fig_name='ADALearning.jpg' )
    plt.show() 
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy:{}".format(Pred))
    menu()

    
def Knn_fun(option):
    print("KNN : {}".format(option))
    t_start = time.clock()
    clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
    clf.fit(X_train,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf.predict(X_test)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of KNN classifier on training set where k=5: {:.2f}'
     .format(clf.score(X_train, y_train)))
    print('Accuracy of KNN Tree classifier on test set where k=5: {:.2f}'
     .format(clf.score(X_test, y_test)))
    t_start = time.clock()
    clf1 = KNeighborsClassifier(n_neighbors=3,metric='minkowski', p=2)
    clf1.fit(X_train,y_train)
    t_end = time.clock()
    p_start = time.clock()
    clf1.predict(X_test)
    p_end = time.clock()
    print ('Training time: %fs' % (t_end - t_start))
    print ('Predicting time: %fs' % (p_end - p_start)) 
    print('Accuracy of KNN classifier on training set where k=3: {:.2f}'
     .format(clf1.score(X_train, y_train)))
    print('Accuracy of KNN Tree classifier on test set where k=3: {:.2f}'
     .format(clf1.score(X_test, y_test)))
    param_range = np.arange(1, 41, 2)
    validation_curve_model(X, y, KNeighborsClassifier(metric='minkowski'),
                           param_name="n_neighbors",parameters=param_range, cv=4, x_label="n_neighbors",fig_name='Knn.jpg')
    plt.show()
    train_size=np.linspace(.1, 1.0, 15)
    Learning_curve_model(X, y, DecisionTreeClassifier(class_weight='balanced'), cv=4, train_sizes=train_size,fig_name='KNnLearning.jpg' )
    plt.show() 
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy of k=5:{}".format(Pred))
    y_pred = clf1.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    Pred = (cm[0,0]+cm[1,1])/6033*100
    print("Prdectied Accuracy of k=3:{}".format(Pred))
    menu()

#Calling Options
def menu():
    print("*****Analysis of 5 Algorithms*****")
    print("Enter the coressponding option for running a model")
    option = int(input("1. DecssionTree 2. Artifical Neural Network 3. SVM With Kernels 4. ADABoosting 5. KNN 6. Exit\nOption:"))
    if option==1:
        decessionTree(option)
    elif option==2:
        ANN_fun(option)
    elif option==3:
        Svm_fun(option)
    elif option==4:
        ADAboost_fun(option)
    elif option==5:
        Knn_fun(option) 
    else:
        print("Exited")
    
    
while True:  
    print("Assignment I")
    menu()
    break
    


    
    




