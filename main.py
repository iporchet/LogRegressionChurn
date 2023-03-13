import sys
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv (r'Churn_Modelling.csv')
cleaned_df = df = df.drop('CustomerId', axis=1)
sns.set_theme(style="dark")


for column in cleaned_df.columns:
    if cleaned_df[column].dtype == np.number:
        continue
    cleaned_df[column] = LabelEncoder().fit_transform(cleaned_df[column])


print("This is cleaned data:\n")
print(cleaned_df.describe())
print(cleaned_df.dtypes)


X = cleaned_df.drop('Exited', axis=1)
y = cleaned_df['Exited']
X = StandardScaler().fit_transform(X)

retained = df[df.Exited == 0]
churned = df[df.Exited == 1]
num_retained = retained.shape[0]
num_churned = churned.shape[0]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(num_retained/(num_retained + num_churned) * 100, "% of customers stayed with the company.")
print("")
print(num_churned / (num_retained + num_churned) * 100,"% of customers left the company.")
print(predictions)
print(classification_report(y_test, predictions))

print(sns.lmplot(x="Age", y="CreditScore", hue="Exited", data=df))
plt.show()
