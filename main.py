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

df = pd.read_csv (r'/home/l3phant/Downloads/Churn_Modelling.csv')

sns.set_theme(style="darkgrid")

print(df.shape)
print(df.columns.values)
print(df.describe())

print(df['Exited'].value_counts())


retained = df[df.Exited == 0]
churned = df[df.Exited == 1]
num_retained = retained.shape[0]
num_churned = churned.shape[0]

print(num_retained/(num_retained + num_churned) * 100, "% of customers stayed with the company.")
print("")
print(num_churned / (num_retained + num_churned) * 100,"% of customers left the company.")


print(sns.lmplot(x="Tenure", y="Age", hue="Exited", data=df))
plt.show()