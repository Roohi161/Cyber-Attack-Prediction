import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv("Dataset/kdd_train.csv",nrows=20000)
labels = np.unique(dataset['labels']).ravel()
#dataset processing like non-numeric data encoding to numeric values and then replace missing values with 0
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):#loop and check each column for non-numeric values
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(0, inplace = True)#replace missing values with 0
#dataset features normalization
Y = dataset['labels'].ravel()#extract attack column as the target class value
Y = Y.astype('int')
dataset.drop(['labels'], axis = 1,inplace=True)#remove irrelevant columns
column_names = dataset.columns
X = dataset.values
X = dataset.values
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle features
X = X[indices]
Y = Y[indices]
scaler = StandardScaler()
X = scaler.fit_transform(X)#normalize training X features
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

rf_cls = RandomForestClassifier()
rf_cls.fit(X_train, y_train)


@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', msg="<font size=3 color=blue>Welcome "+user+"</font>")
        else:
            return render_template('UserLogin.html', msg="<font size=3 color=red>Invalid login details</font>")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global scaler, dataset, labels, optimized_model, label_encoder, rf_cls
        testData = pd.read_csv("Dataset/testData.csv")#load test data
        data = testData.values
        for i in range(len(label_encoder)-1):#label encoding from non-numeric to numeric
            le = label_encoder[i]
            testData[le[0]] = pd.Series(le[1].transform(testData[le[0]].astype(str)))#encode all str columns to numeric
        testData.fillna(0, inplace = True)#replace misisng values with mean    
        testData = testData.values    
        testData = scaler.transform(testData)#normalize test data
        predict = rf_cls.predict(testData)#apply extension hybrid model to predict attack type
        output = '<table border=1 align=center width=100%><tr>'
        output += '<th><font size="3" color="black">Test Data</font></th>'
        output += '<th><font size="3" color="blue">Predicted Cyber Attack</font></th></tr>'
        for i in range(len(predict)):
            output += "<tr>"
            output += '<td><font size="3" color="black">'+str(data[i])+'</font></td>'
            if labels[predict[i]] == "normal":
                output += '<td><font size="4" color="green">'+labels[predict[i]]+'</font></td>'
            else:
                output += '<td><font size="4" color="red">'+labels[predict[i]]+'</font></td>'    
        output += "</table><br/><br/><br/><br/>"             
        return render_template('UserScreen.html', msg=output)

if __name__ == '__main__':
    app.run()    
