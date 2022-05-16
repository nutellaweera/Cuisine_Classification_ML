import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}

x_ingredients, y_cuisines = utils.get_formatted_data()
x_train, x_test, y_train, y_test = train_test_split(x_ingredients, y_cuisines, random_state=42)

# grid search with 3 fold cross validation for the pre-defined alpha values
nb_grid = GridSearchCV(MultinomialNB(), param_grid=params,cv=3,verbose=4)
nb_grid.fit(x_train, y_train)

print(f'Train score: {nb_grid.best_estimator_.score(x_train, y_train)}')
print(f'Test score: {nb_grid.best_estimator_.score(x_test, y_test)}')

print(f'Best accuracy: {nb_grid.best_score_}')
print(f'Best params: {nb_grid.best_params_}')

# Output: (alpha of 0.01 was selected)
# Fitting 3 folds for each of 5 candidates, totalling 15 fits
# [CV 1/3] END ........................alpha=0.01;, score=0.738 total time=   0.1s
# [CV 2/3] END ........................alpha=0.01;, score=0.737 total time=   0.1s
# [CV 3/3] END ........................alpha=0.01;, score=0.730 total time=   0.1s
# [CV 1/3] END .........................alpha=0.1;, score=0.735 total time=   0.1s
# [CV 2/3] END .........................alpha=0.1;, score=0.733 total time=   0.1s
# [CV 3/3] END .........................alpha=0.1;, score=0.730 total time=   0.1s
# [CV 1/3] END .........................alpha=0.5;, score=0.671 total time=   0.1s
# [CV 2/3] END .........................alpha=0.5;, score=0.663 total time=   0.1s
# [CV 3/3] END .........................alpha=0.5;, score=0.665 total time=   0.1s
# [CV 1/3] END .........................alpha=1.0;, score=0.625 total time=   0.1s
# [CV 2/3] END .........................alpha=1.0;, score=0.624 total time=   0.1s
# [CV 3/3] END .........................alpha=1.0;, score=0.627 total time=   0.1s
# [CV 1/3] END ........................alpha=10.0;, score=0.496 total time=   0.1s
# [CV 2/3] END ........................alpha=10.0;, score=0.497 total time=   0.1s
# [CV 3/3] END ........................alpha=10.0;, score=0.495 total time=   0.1s
# Train score: 0.8488434461951055
# Test score: 0.745575221238938
# Best accuracy: 0.7351658509551641
# Best params: {'alpha': 0.01}