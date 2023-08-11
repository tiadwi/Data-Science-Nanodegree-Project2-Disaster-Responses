import sys
import pandas as pd
import numpy as np
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    ''' This function loads and merges two data from different filepaths
    
    input: 
    message_filepath: filepath for message.csv
    categories_filepath: filepath for categories.csv

    output:
    a dataframe containing both merged dataset
    '''
    # Load dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge dataset on id and assign to df
    df = pd.merge(messages, categories, on='id', how='outer')
    return df

def clean_data(df):
    ''' This function perform data cleaning for df dataframe
    
    input: 
    df: raw dataframe
    
    output:
    df: cleaned dataframe
    '''
    #split categories into separate categories
    categories = df['categories'].str.split(";", expand=True)
    # select the first row to extract categories name
    row = categories.iloc[0]
    # apply lambda function to create new column names for categories
    category_colnames = list(map(lambda x: x.split("-")[0], categories.iloc[0].values.tolist()))
    # rename the categories column
    categories.columns = category_colnames

    # iterate through the category columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    # drop the original categories column from 'df'    
    del df['categories']
    #concantenate the original dataframe with the new dataframe
    df = pd.concat([df, categories], axis=1)
    #drop duplicate
    df = df.drop_duplicates(keep='first')
    return df


def save_data(df, database_filename):
    '''save the cleaned dataframe into a given database
    input: 
    df dataframe
    database_filename
    '''
    engine = create_engine('sqlite:///'+database_filename)
    #name the database 'DisasterResponse'
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()