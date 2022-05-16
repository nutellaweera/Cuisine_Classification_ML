import pandas as pd
from nltk import WordNetLemmatizer
import inflect
from sklearn.feature_extraction.text import TfidfVectorizer

wnl = WordNetLemmatizer()
inf = inflect.engine()

# Pre-process ingredient lists: 
#   1. Convert to lowercase 
#   2. Strip leading/trailing whitespace
#   3. Remove non alpha (numbers/punctuation) characters
#   4. Singularize the words 
#   5. Lemmatize 
# And return in comma-seperated string form
def format_ingredients(ingredient_list):
    formatted = [ing.strip().lower() for ing in ingredient_list]
    alpha = [(''.join(char for char in ing if char.isalpha())) for ing in formatted]
    singular = [inf.singular_noun(ing) or ing for ing in alpha]
    lemmatized = [wnl.lemmatize(ing) for ing in singular]
    return (', '.join(lemmatized))

# Read the json dataset into a pandas dataframe,
# Add a new ingredients_formatted column to the df (with pre-processed ingredients)
# Vectorize the ingredient strings with tf-idf
# Return the x and y params where
#   x -> vectorized ingredients
#   y -> cuisine type
def get_formatted_data():
    df = pd.read_json('dataset.json')
    df['ingredients_formatted'] = df['ingredients'].apply(lambda x: format_ingredients(x))
    tfidf = TfidfVectorizer(stop_words='english', analyzer='word', max_df=0.8, token_pattern=r'\w+')
    x = tfidf.fit_transform(df['ingredients_formatted'])
    y = df['cuisine']
    return x,y