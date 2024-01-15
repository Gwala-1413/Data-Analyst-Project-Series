import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statistics as sts

#Load Iris.csv into a pandas dataFrame.
iris = pd.read_csv("Iris.csv")

iris = iris.drop('Id', axis=1)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print()

# Get First 5 features
print("\t\t\tTOP 5 READINGS \n")
print(iris.head(),"\n")
print("-----------------------------------------------------------------------------------------------------------------")

# Get Last 5 features
print("\t\t\tLAST 5 READINGS \n")
print(iris.tail(),"\n")
print("-----------------------------------------------------------------------------------------------------------------")

# Get info about data content
print(iris.info(),"\n")
print("-----------------------------------------------------------------------------------------------------------------")

# Check for duplicate values
print("CHECK FOR DUPLICATE VALUES \n")
print(iris.duplicated(),"\n")
print("-----------------------------------------------------------------------------------------------------------------")

# Check for null values
print("CHECK FOR NULL VALUES \n")
print(iris.isnull().sum(),"\n")
print("-----------------------------------------------------------------------------------------------------------------")

# Do the statistics on the data file
print("\t\t\t STATS \n")
print(iris.describe())
print("mode \t  ",sts.mode(iris['sepal_length']),"\t       ",sts.mode(iris['sepal_width']),"\t     ",sts.mode(iris['petal_length']),"\t  ",sts.mode(iris['petal_width']))
print()
print("-----------------------------------------------------------------------------------------------------------------")

# Get Row by Column count
print("ROW BY COLUMN")
print(iris.shape)
print()
print("-----------------------------------------------------------------------------------------------------------------")

# Get Column names
print("\t\t COLUMN NAMES \n")
print(iris.columns)
print()
print("-----------------------------------------------------------------------------------------------------------------")

# Contents of Species column
print("\t CONTENT COUNT OF SPECIES COLUMN\n")
print(iris["species"].value_counts())
print()
print("-----------------------------------------------------------------------------------------------------------------")

sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=6) \
   .map(plt.scatter, "petal_length", "petal_width") \
   .add_legend();
plt.show();

sns.FacetGrid(iris, hue="species", height=5) \
   .map(sns.histplot, "sepal_length") \
   .add_legend();
plt.show();

# Plot a graph for the columns in the file
sns.set_style('darkgrid')
sns.pairplot(iris,hue='species',height=2,)
plt.show()

# Get distribution plot for petal length
print("\n")
sns.FacetGrid(iris, hue="species", height=5) \
   .map(sns.distplot, "petal_length") \
   .add_legend()
plt.show()

# Get distribution plot for petal width
print("\n")
sns.FacetGrid(iris, hue="species", height=5) \
   .map(sns.distplot, "petal_width") \
   .add_legend();
plt.show();

iris.hist(edgecolor='black',bins = 20, figsize= (12,6))
plt.show()
