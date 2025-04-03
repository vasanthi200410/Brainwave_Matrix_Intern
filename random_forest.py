# -*- coding: utf-8 -*-
"""random-forest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ojWyDBAH6TLPu6PGYBgjKklDSclrRcmY
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
data = pd.read_csv("/content/creditcard.csv")

# Start exploring the dataset
print(data.columns)
data.head()

# Print the shape of the data
# data = data.sample(frac=0.1, random_state = 48)
print(data.shape)
print(data.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

# distribution of Amount
amount = [data['Amount'].values]
sns.distplot(amount)

# distribution of Time
time = data['Time'].values
sns.distplot(time)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Ensure 'data' is loaded
data = pd.read_csv("/content/creditcard.csv")  # Replace with actual file path

# Select first 28 columns
features = data.iloc[:, 0:5].columns

plt.figure(figsize=(7, 5 * 4))
gs = gridspec.GridSpec(5, 1)

for i, c in enumerate(features):
    ax = plt.subplot(gs[i])
    sns.histplot(data=data, x=c, hue="Class", bins=50, kde=True, element="step")  # Updated for distplot deprecation
    ax.set_xlabel('')
    ax.set_title(f'Histogram of feature: {c}')

plt.tight_layout()
plt.show()

# Plot histograms of each parameter
data.hist(figsize = (20, 20))
plt.show()

# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print("Amount details of fradulent transacation")
Fraud.Amount.describe()

print("Amount details of valid transaction")
Valid.Amount.describe()

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (9, 6))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#seperating the X and the Y from the dataset
X=data.drop(['Class'], axis=1)
Y=data["Class"]
print(X.shape)
print(Y.shape)
#getting just the values for the sake of processing (its a numpy array with no columns)
X_data=X.values
Y_data=Y.values

X_data

"""## Using Scikit-learn to split data into training and testing sets"""

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

"""## Random Forest Classifier"""

# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
# predictions
y_pred = rfc.predict(X_test)

#Evaluating the classifier
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
n_errors = (y_pred != Y_test).sum()
print("The model used is Random Forest classifier")
acc= accuracy_score(Y_test,y_pred)
print("The accuracy is  {}".format(acc))
prec= precision_score(Y_test,y_pred)
print("The precision is {}".format(prec))
rec= recall_score(Y_test,y_pred)
print("The recall is {}".format(rec))
f1= f1_score(Y_test,y_pred)
print("The F1-Score is {}".format(f1))
MCC=matthews_corrcoef(Y_test,y_pred)
print("The Matthews correlation coefficient is {}".format(MCC))


#printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))