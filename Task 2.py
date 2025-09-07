# Titanic Dataset - Data Cleaning and EDA

#  Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Load Dataset
df = sns.load_dataset('titanic')  # If using Kaggle CSV: pd.read_csv('titanic.csv')

#  Basic Info
print("Basic Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst 5 Rows:\n", df.head())

#  Data Cleaning

# Drop columns with too many missing values
df.drop(['deck'], axis=1, inplace=True)

# Fill missing age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill embarked with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop rows with missing embark_town
df.dropna(subset=['embark_town'], inplace=True)

#  Exploratory Data Analysis (EDA)

# 1 Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title('Survival Counts')
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.xlabel('Survival')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2 Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 3 Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 4 Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()

# 5 Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()