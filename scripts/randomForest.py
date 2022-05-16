import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_ingredients, y_cuisines = utils.get_formatted_data()
x_train, x_test, y_train, y_test = train_test_split(x_ingredients, y_cuisines, random_state=42)

clf_ovr = LogisticRegression(max_iter=50, multi_class='ovr')
clf_ovr.fit(x_train, y_train)
print(f'Train score: {clf_ovr.score(x_train, y_train)}') # 81.37%
print(f'Test score: {clf_ovr.score(x_test, y_test)}') # 76.56%

clf_multi = LogisticRegression(max_iter=50, multi_class='multinomial')
clf_multi.fit(x_train, y_train)
print(f'Train score: {clf_multi.score(x_train, y_train)}') # 81.52%
print(f'Test score: {clf_multi.score(x_test, y_test)}') # 76.78%