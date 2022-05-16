import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x_ingredients, y_cuisines = utils.get_formatted_data()
x_train, x_test, y_train, y_test = train_test_split(x_ingredients, y_cuisines, random_state=42)

# trying out default params to get a ball-park of expected scores
# clf = KNeighborsClassifier()
# clf.fit(x_train, y_train)
# print(f'Train score: {clf.score(x_train, y_train)}') #81.29%
# print (f'Test score: {clf.score(x_test, y_test)}') # 71.33% not bad!

# Fits a knn classifier for the preset x, y based on the specified range,
# returns a dataframe of test and train scores for plotting
def fit_and_plot(start, end, step, distance_metric):
    print(f'{start} to {end} in steps of {step} with {distance_metric} distance...')
    scores = []
    for k in range(start, end, step):
        clf = KNeighborsClassifier(k, metric=distance_metric)
        clf.fit(x_train, y_train)
        scores.append([k, distance_metric, clf.score(x_train, y_train), clf.score(x_test, y_test)])
    scores = pd.DataFrame(scores, columns=['Neigbours', 'Distance', 'Training', 'Testing'])
    print(scores)
    return scores

# plot the accuracy of a combination of k values and distance metrics
distance_metrics = ['euclidean','manhattan']
start, end, step = 1, 27, 2

for metric in distance_metrics:
    df = fit_and_plot(start, end, step, metric)
    plt.plot(range(start, end, step), df['Training'], label=f'Training_{metric}')
    plt.plot(range(start, end, step), df['Testing'], label=f'Testing_{metric}')
plt.ylabel('Accuracy')
plt.xlabel('Num neigbours')
plt.legend()
plt.savefig(f'graphs_and_vis/knn')
    
# based on the above, euclidean distance with 17 neigbours was selected.
# scores: train -> 0.776634, test -> 0.743564




