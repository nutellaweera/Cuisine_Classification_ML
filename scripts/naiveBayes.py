import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x_ingredients, y_cuisines = utils.get_formatted_data()
x_train, x_test, y_train, y_test = train_test_split(x_ingredients, y_cuisines, random_state=42)


clf = KNeighborsClassifier(metric='hamming')
clf.fit(x_train, y_train)
print(f'Train score: {clf.score(x_train, y_train)}') #81.29%
print (f'Test score: {clf.score(x_test, y_test)}') # 71.33% not bad!