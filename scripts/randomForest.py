import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

x_ingredients, y_cuisines = utils.get_formatted_data()
x_train, x_test, y_train, y_test = train_test_split(x_ingredients, y_cuisines, random_state=42)

params = {
    'n_estimators': [1, 10, 100, 200], # 1 = dt
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 8, 50],
    }

# grid search with 3 fold cross validation
rf_grid = GridSearchCV(RandomForestClassifier(bootstrap=True), param_grid=params, cv=3, verbose=4)
rf_grid.fit(x_train, y_train)

print(f'Train score: {rf_grid.best_estimator_.score(x_train, y_train)}')
print(f'Test score: {rf_grid.best_estimator_.score(x_test, y_test)}')

print(f'Best accuracy: {rf_grid.best_score_}')
print(f'Best params: {rf_grid.best_params_}')

# Output:
# Fitting 3 folds for each of 24 candidates, totalling 72 fits
# [CV 1/3] END criterion=gini, min_samples_split=2, n_estimators=1;, score=0.472 total time=   0.3s
# [CV 2/3] END criterion=gini, min_samples_split=2, n_estimators=1;, score=0.442 total time=   0.2s
# [CV 3/3] END criterion=gini, min_samples_split=2, n_estimators=1;, score=0.450 total time=   0.2s
# [CV 1/3] END criterion=gini, min_samples_split=2, n_estimators=10;, score=0.653 total time=   2.2s
# [CV 2/3] END criterion=gini, min_samples_split=2, n_estimators=10;, score=0.645 total time=   2.2s
# [CV 3/3] END criterion=gini, min_samples_split=2, n_estimators=10;, score=0.648 total time=   2.2s
# [CV 1/3] END criterion=gini, min_samples_split=2, n_estimators=100;, score=0.693 total time=  21.6s
# [CV 2/3] END criterion=gini, min_samples_split=2, n_estimators=100;, score=0.689 total time=  21.6s
# [CV 3/3] END criterion=gini, min_samples_split=2, n_estimators=100;, score=0.691 total time=  21.7s
# [CV 1/3] END criterion=gini, min_samples_split=2, n_estimators=200;, score=0.698 total time=  43.1s
# [CV 2/3] END criterion=gini, min_samples_split=2, n_estimators=200;, score=0.691 total time=  43.0s
# [CV 3/3] END criterion=gini, min_samples_split=2, n_estimators=200;, score=0.695 total time=  42.9s
# [CV 1/3] END criterion=gini, min_samples_split=8, n_estimators=1;, score=0.476 total time=   0.2s
# [CV 2/3] END criterion=gini, min_samples_split=8, n_estimators=1;, score=0.465 total time=   0.2s
# [CV 3/3] END criterion=gini, min_samples_split=8, n_estimators=1;, score=0.465 total time=   0.2s
# [CV 1/3] END criterion=gini, min_samples_split=8, n_estimators=10;, score=0.662 total time=   1.6s
# [CV 2/3] END criterion=gini, min_samples_split=8, n_estimators=10;, score=0.651 total time=   1.6s
# [CV 3/3] END criterion=gini, min_samples_split=8, n_estimators=10;, score=0.657 total time=   1.6s
# [CV 1/3] END criterion=gini, min_samples_split=8, n_estimators=100;, score=0.694 total time=  15.7s
# [CV 2/3] END criterion=gini, min_samples_split=8, n_estimators=100;, score=0.687 total time=  15.6s
# [CV 3/3] END criterion=gini, min_samples_split=8, n_estimators=100;, score=0.687 total time=  15.8s
# [CV 1/3] END criterion=gini, min_samples_split=8, n_estimators=200;, score=0.694 total time=  31.3s
# [CV 2/3] END criterion=gini, min_samples_split=8, n_estimators=200;, score=0.692 total time=  31.7s
# [CV 3/3] END criterion=gini, min_samples_split=8, n_estimators=200;, score=0.687 total time=  31.3s
# [CV 1/3] END criterion=gini, min_samples_split=50, n_estimators=1;, score=0.487 total time=   0.2s
# [CV 2/3] END criterion=gini, min_samples_split=50, n_estimators=1;, score=0.477 total time=   0.1s
# [CV 3/3] END criterion=gini, min_samples_split=50, n_estimators=1;, score=0.470 total time=   0.1s
# [CV 1/3] END criterion=gini, min_samples_split=50, n_estimators=10;, score=0.652 total time=   1.1s
# [CV 2/3] END criterion=gini, min_samples_split=50, n_estimators=10;, score=0.654 total time=   1.2s
# [CV 3/3] END criterion=gini, min_samples_split=50, n_estimators=10;, score=0.652 total time=   1.1s
# [CV 1/3] END criterion=gini, min_samples_split=50, n_estimators=100;, score=0.678 total time=  11.1s
# [CV 2/3] END criterion=gini, min_samples_split=50, n_estimators=100;, score=0.670 total time=  11.1s
# [CV 3/3] END criterion=gini, min_samples_split=50, n_estimators=100;, score=0.671 total time=  11.2s
# [CV 1/3] END criterion=gini, min_samples_split=50, n_estimators=200;, score=0.679 total time=  22.1s
# [CV 2/3] END criterion=gini, min_samples_split=50, n_estimators=200;, score=0.673 total time=  22.5s
# [CV 3/3] END criterion=gini, min_samples_split=50, n_estimators=200;, score=0.675 total time=  22.3s
# [CV 1/3] END criterion=entropy, min_samples_split=2, n_estimators=1;, score=0.402 total time=   0.3s
# [CV 2/3] END criterion=entropy, min_samples_split=2, n_estimators=1;, score=0.419 total time=   0.3s
# [CV 3/3] END criterion=entropy, min_samples_split=2, n_estimators=1;, score=0.434 total time=   0.3s
# [CV 1/3] END criterion=entropy, min_samples_split=2, n_estimators=10;, score=0.606 total time=   2.4s
# [CV 2/3] END criterion=entropy, min_samples_split=2, n_estimators=10;, score=0.606 total time=   2.4s
# [CV 3/3] END criterion=entropy, min_samples_split=2, n_estimators=10;, score=0.601 total time=   2.4s
# [CV 1/3] END criterion=entropy, min_samples_split=2, n_estimators=100;, score=0.666 total time=  24.0s
# [CV 2/3] END criterion=entropy, min_samples_split=2, n_estimators=100;, score=0.661 total time=  23.7s
# [CV 3/3] END criterion=entropy, min_samples_split=2, n_estimators=100;, score=0.663 total time=  23.9s
# [CV 1/3] END criterion=entropy, min_samples_split=2, n_estimators=200;, score=0.667 total time=  47.7s
# [CV 2/3] END criterion=entropy, min_samples_split=2, n_estimators=200;, score=0.663 total time=  47.8s
# [CV 3/3] END criterion=entropy, min_samples_split=2, n_estimators=200;, score=0.663 total time=  47.7s
# [CV 1/3] END criterion=entropy, min_samples_split=8, n_estimators=1;, score=0.445 total time=   0.2s
# [CV 2/3] END criterion=entropy, min_samples_split=8, n_estimators=1;, score=0.414 total time=   0.2s
# [CV 3/3] END criterion=entropy, min_samples_split=8, n_estimators=1;, score=0.439 total time=   0.2s
# [CV 1/3] END criterion=entropy, min_samples_split=8, n_estimators=10;, score=0.619 total time=   1.6s
# [CV 2/3] END criterion=entropy, min_samples_split=8, n_estimators=10;, score=0.620 total time=   1.6s
# [CV 3/3] END criterion=entropy, min_samples_split=8, n_estimators=10;, score=0.623 total time=   1.6s
# [CV 1/3] END criterion=entropy, min_samples_split=8, n_estimators=100;, score=0.662 total time=  15.9s
# [CV 2/3] END criterion=entropy, min_samples_split=8, n_estimators=100;, score=0.656 total time=  15.8s
# [CV 3/3] END criterion=entropy, min_samples_split=8, n_estimators=100;, score=0.655 total time=  15.6s
# [CV 1/3] END criterion=entropy, min_samples_split=8, n_estimators=200;, score=0.660 total time=  31.7s
# [CV 2/3] END criterion=entropy, min_samples_split=8, n_estimators=200;, score=0.659 total time=  31.6s
# [CV 3/3] END criterion=entropy, min_samples_split=8, n_estimators=200;, score=0.657 total time=  31.1s
# [CV 1/3] END criterion=entropy, min_samples_split=50, n_estimators=1;, score=0.434 total time=   0.1s
# [CV 2/3] END criterion=entropy, min_samples_split=50, n_estimators=1;, score=0.451 total time=   0.1s
# [CV 3/3] END criterion=entropy, min_samples_split=50, n_estimators=1;, score=0.459 total time=   0.1s
# [CV 1/3] END criterion=entropy, min_samples_split=50, n_estimators=10;, score=0.603 total time=   1.0s
# [CV 2/3] END criterion=entropy, min_samples_split=50, n_estimators=10;, score=0.600 total time=   1.0s
# [CV 3/3] END criterion=entropy, min_samples_split=50, n_estimators=10;, score=0.605 total time=   1.1s
# [CV 1/3] END criterion=entropy, min_samples_split=50, n_estimators=100;, score=0.632 total time=  10.2s
# [CV 2/3] END criterion=entropy, min_samples_split=50, n_estimators=100;, score=0.626 total time=  10.2s
# [CV 3/3] END criterion=entropy, min_samples_split=50, n_estimators=100;, score=0.630 total time=  10.1s
# [CV 1/3] END criterion=entropy, min_samples_split=50, n_estimators=200;, score=0.633 total time=  20.2s
# [CV 2/3] END criterion=entropy, min_samples_split=50, n_estimators=200;, score=0.628 total time=  20.3s
# [CV 3/3] END criterion=entropy, min_samples_split=50, n_estimators=200;, score=0.631 total time=  20.2s
# Train score: 0.999597720415689
# Test score: 0.7153057119871279
# Best accuracy: 0.6947367357844244
# Best params: {'criterion': 'gini', 'min_samples_split': 2, 'n_estimators': 200}