import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print("🎉 All required libraries are installed and working perfectly!")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("Seaborn:", sns.__version__)

df = pd.read_excel(r"C:\AI Search Trend Analysis & Insight Dashboard\EDA\AI_Dataset_Professional.xlsx")
print(df.head())

print("Shape (Rows, Columns):", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
# Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())
# Handle Missing values
df.fillna(0, inplace=True)
#Basic Data Understanding
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
#Univariate Analysis
#Numerical Columns
df['Search Interest'].describe()
df['Growth %'].describe()
sns.histplot(df['Search Interest'], kde=True)
plt.title("Search Interest Distribution")
plt.show()
sns.histplot(df['Growth %'], kde=True)
plt.title("Growth % Distribution")
plt.show()
#Categorical Columns
df['Category'].value_counts()
df['Platform'].value_counts()
df['Trend Status'].value_counts()
df['Popularity Level'].value_counts()
sns.countplot(x='Category', data=df)
plt.xticks(rotation=30)
plt.title("Category Distribution")
plt.show()
#Bivariate Analysis
#Category vs Search Interest
df.groupby('Category')['Search Interest'].mean().sort_values(ascending=False)
sns.barplot(x='Category', y='Search Interest', data=df)
plt.xticks(rotation=30)
plt.title("Category-wise Search Interest")
plt.show()
#Platform vs Search Interest
df.groupby('Platform')['Search Interest'].mean().sort_values(ascending=False)
sns.barplot(x='Platform', y='Search Interest', data=df)
plt.title("Platform-wise Search Interest")
plt.show()
#Trend Status vs Growth %
df.groupby('Trend Status')['Growth %'].mean()
sns.boxplot(x='Trend Status', y='Growth %', data=df)
plt.title("Trend Status vs Growth %")
plt.show()
#Multivariate Analysis
#Correlation Heatmap
sns.heatmap(df[['Search Interest','Growth %']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
#Key Insights in this Dataset
#Most Popular AI Keywords
df.sort_values(by='Search Interest', ascending=False).head(10)
#Most Trending AI Category
df.groupby('Category')['Search Interest'].mean().sort_values(ascending=False)
#Platform Dominance
df.groupby('Platform')['Search Interest'].mean().sort_values(ascending=False)
#Growth Trend Patterns
df['Trend Status'].value_counts()
#High Growth Opportunities
df.sort_values(by='Growth %', ascending=False).head(5)