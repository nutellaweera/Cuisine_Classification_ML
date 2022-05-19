# Cuisine_Classification_ML

Code and artefacts related to an attempt to identify the cuisine type of a recipe based on its ingredients.

## Dataset
[Kaggle-Whats cooking?](https://www.kaggle.com/competitions/whats-cooking/data)

## Repository structure
* Pre-processing and EDA - [Jupyter notebook](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/preprocessingAndEDA.ipynb)
* Training and testing - [Jupyter notebook](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/trainingAndTesting.ipynb) 
* EDA with Pandas Profiling (hosted on Pages) - https://nutellaweera.github.io/Cuisine_Classification_ML/
* Exploratory scripts (experimentation with parameters, tuning using grid search/trial-and-error) - [/scripts](https://github.com/nutellaweera/Cuisine_Classification_ML/tree/main/scripts) folder.
  * [Helper/utility methods](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/utils.py)
  * [KNN](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/kNN.py)
  * [Multiclass Logistic Regression](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/multiclassLogisticRegression.py)
  * [Naive Bayes](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/naiveBayes.py)
  * [Random Forest](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/randomForest.py)
  * [K-means Clustering](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/kMeans_clustering.py)
* Graphs and visualizations in the [/graphs_and_vis](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/scripts/kMeans_clustering.py) folder

## Visualizations and results
### Model parameters and performance (avg accuracy scores):

![perf_table](https://user-images.githubusercontent.com/8774488/169342016-1e497562-b505-4543-8b53-1b355602e633.png)

![perf_boxplot](graphs_and_vis/algo_performance.jpeg)

### EDA and exploration
Ingredient counts:

![ing_counts](graphs_and_vis/cuisine_counts.png)

Word cloud:

![word_cloud](graphs_and_vis/ingredient_wordcloud.png)


