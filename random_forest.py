from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
y = pd.factorize(train['species'])[0]
actual = pd.factorize(test['species'])[0]

seed = 0
kfold = model_selection.KFold(n_splits=8, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, train[features], y, cv=kfold)
print(results.mean())


model.fit(train[features], y)
predicted = model.predict(test[features])
print(predicted)
print(actual)
# clf = RandomForestClassifier(n_jobs=2)

# clf.predict(test[features])


# preds = iris.target_names[clf.predict(test[features])]

# pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])


# print(list(zip(train[features], clf.feature_importances_)))