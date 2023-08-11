import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    ''' 
    This function loads data from given database path

    Input: database_filepath (a filepath to the database)
    Output: X as features, Y as target, and category names

    '''
    #load data from database 
    engine = create_engine('sqlite:///'+ database_filepath)
    df= pd.read_sql_table('messages', engine)

    #define features and target
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Removes whitespaces, reduces each word to its base form, converts to lowercase and return an array of each word
    in a message
    '''
     # Converting everything to lower case
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
  
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function builds a machine learning pipeline to train and 
    uses Grid Search to find the best parameters.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #setting up parameters
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function calculates the classification reports for each of category names
    '''
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(class_report)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterCleaned.db classifier.pkl')


if __name__ == '__main__':
    main()