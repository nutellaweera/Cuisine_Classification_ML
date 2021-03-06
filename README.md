# Cuisine_Classification_ML

Contains code and artefacts used to attempt identification of the cuisine type of a recipe based on its ingredients, as part of the coursework for the Machine Learning and Predictive Analytics Module for UWE MSc in Data Science.

## Dataset
Kaggle - [Whats cooking?](https://www.kaggle.com/competitions/whats-cooking/data)

## Repository structure
* Pre-processing and EDA - [Jupyter notebook](preprocessingAndEDA.ipynb)
* Training and testing - [Jupyter notebook](trainingAndTesting.ipynb) 
* EDA with Pandas Profiling (hosted on Pages) - https://nutellaweera.github.io/Cuisine_Classification_ML/
* Exploratory scripts (experimentation with parameters, tuning using grid search/trial-and-error) - [/scripts](https://github.com/nutellaweera/Cuisine_Classification_ML/tree/main/scripts) folder.
  * [Helper/utility methods](scripts/utils.py)
  * [KNN](scripts/kNN.py)
  * [Naive Bayes](scripts/naiveBayes.py)
  * [Multiclass Logistic Regression](scripts/multiclassLogisticRegression.py)
  * [Random Forest](scripts/randomForest.py)
  * [K-means Clustering](scripts/kMeans_clustering.py)
* Graphs and visualizations in the [/graphs_and_vis](https://github.com/nutellaweera/Cuisine_Classification_ML/blob/main/graphs_and_vis) folder

## Visualizations and results
### Model parameters and performance (avg accuracy scores):
![perf_table](https://user-images.githubusercontent.com/8774488/169342016-1e497562-b505-4543-8b53-1b355602e633.png)

![perf_boxplot](graphs_and_vis/algo_performance.jpeg)

### EDA and exploration
Ingredient counts:

![ing_counts](graphs_and_vis/cuisine_counts.png)

Word cloud:

![word_cloud](graphs_and_vis/ingredient_wordcloud.png)

### Model training and testing
KNN scores:

![knn_scores](graphs_and_vis/knn.png)


Ingredient clusters (k-means):

![ing_clusters](https://user-images.githubusercontent.com/8774488/169344263-cbdde599-ce5d-43fb-998c-349b8630d733.png)

Confusion matrix:

![conf_m](graphs_and_vis/confusion_matrix.jpeg)



