# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)


df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

features = df.columns[:4]

y = pd.factorize(train['species'])[0]

# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features], y)

clf.predict(test[features])


preds = iris.target_names[clf.predict(test[features])]

pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])


print(list(zip(train[features], clf.feature_importances_)))