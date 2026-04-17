##
# Warner Nash
# 12/04/26
#stellar classification dataset
# superivised multi classification model that classifies celestial objects as either Stars, galaxies or QSO


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score , classification_report
from sklearn.impute import SimpleImputer

df = pd.read_csv('star_classification.csv') # converting the dataset into a pandas dataframe


#getting useful in formation to inform my next steps
print(df.shape)#number of rows and columnns
print(df.head())#first 5 rows
print(df.info())#data types andd null counts
print(df.describe()) #min max and mean of each column
print(df['class'].value_counts()) #how many galaxy qso and start samples there are


#dropping irrelevant columns

#dropped columnm reasoning: obj_id acts as row number, spec_obj_Id: another ID , run_ID: which camera captured the object
#rerun_id: which pipeline was used, cam_col: camera column that captured it, field_id: which feilds scan it was cauight
#plate: the sspectrocopic plate that was used, MJD: the date it was obeserved, fiber_ID: whcih fiber optic cable carried the light

#the stated columns contain information on how or when the object was observed/recorded not what the object is
# a quasar would be a quasar regardless of the date or equipment used to obsertve it

dropped_columns = ['obj_ID' , 'run_ID', 'rerun_ID', 'cam_col','field_ID',
                   'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']

#remaining columnns:
#u,g,r,i,z: brightness of th object recorded in different wavelengths
#alpha,delta: #coordinates
#redshift an attribute of the object that indicates its lighjt is shifted toward longer redder wavelenghts
#due to the expanision of the universe or it travelling further away from earth


#removing desired columns
df = df.drop(columns=dropped_columns)

#printing to check everything is how i want it to be
print(df.shape)
print(df.head())

print(df.isnull().sum()) #checking for missing values

print(df.describe())#checking for out of rangge values
#returned -9999.0000000 after research determined this is a placeholder/sentinel value

#going to check how many rows have a sentinel value
filter_cols = ["u", "g", "r", "i", "z"]
mask = (df[filter_cols] < -9998).any(axis=1) # initial method didnt work so tried this it creates a condtiion and assigns a boolean to each roow
#print((df[filter_cols] == -9999).sum())
#print('number of rows containing a sentinel value:', (df[filter_cols] == 9999).any(axis=1).sum())
print('number of rows containing a sentinel value:', mask.sum())
print(df[mask])

df = df[~mask] #removes the row with a ture value (79543)
print(df.shape)# checking the row was removed

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"]) #using label encoder rather than map so we can decode later on
#label encoder is also more dynamic and useful if we have many classes

print(df['class'].value_counts())
print(le.classes_)


#seperating features and targets

x = df.drop(columns=['class']) #features
y = df['class'] # targets

print(x.shape)
print(y.shape)
#print check

#scaling and normalizes features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(x_scaled.mean(axis=0).round(2))
print(x_scaled.std(axis=0).round(2))

#test train split
#80 train 20 test
#random state 42 so split stays consistent
#stratify y to ensure class proportions remmain the saem in train and test spltis

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42, stratify = y)

print(x_test.shape)
print(x_train.shape)

#knn classifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
#finds 5 similar objects in the dataset and picks whatever class the majority belong to
knn.fit(x_train, y_train)
# target variarbles
print("knn done")

#decision tree
#learns a set of rules to determine a class by its charateristics
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, random_state = 42)
dt.fit(x_train, y_train)
print("decision tree done")

#support vector machine(SVM) training
svm = SVC(kernel = 'rbf', gamma = 'auto')
svm.fit(x_train, y_train)
print("svm done")

#Logistical regression training
lr = LogisticRegression(max_iter=1000, solver='lbfgs')
lr.fit(x_train, y_train)
print("Logistic Regression trained")


#getting predictios from all four models
y_pred_knn = knn.predict(x_test)
y_pred_lr = lr.predict(x_test)
y_pred_dt = dt.predict(x_test)
y_pred_svm = svm.predict(x_test)

#plotting the confusion matrices
fig, axes = plt.subplots(2,2, figsize=(12,10))

models = [('KNN', y_pred_knn), ('logistic regression', y_pred_lr),
          ('SVM', y_pred_svm), ('decision tree', y_pred_dt) ]

#confusion matrix
#confusion matrix is a table that shows the number of correct and incorrect predictions made by the model
for ax, (name, y_pred) in zip(axes.flatten(), models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)#annot true adds the numbers to the heatmap fmt d formats the numbers as integers, cmap blues gives it a blue color scheme, xticklabels and yticklabels label the axes with the class names
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('actual')
plt.tight_layout()#
plt.savefig('cm.png')
plt.show()

#setting up a bar chart to compare the f1 scores of the four models
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

#
models = ['KNN', 'Logistic Regression', 'Decision Tree', 'SVM']
f1_scores = [f1_knn, f1_lr, f1_dt, f1_svm]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color=['#2196F3','#4CAF50', '#FF9800', '#E91E63'])
plt.ylim(0.8, 1.0)
plt.title('F1 Score Comparison Across Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Weighted F1 Score')

for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('f1_comparison.png')
plt.show()
    

