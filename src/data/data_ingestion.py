import pandas as pd
import numpy as np
import time
import logging
import os
import dotenv
from transformers import pipeline

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_ingestion.log',
                    filemode='a')
CURRENT_YEAR = time.localtime().tm_year



def ingest_data():
    """Load raw book data from CSV file."""
    try:
        file_path = 'data/raw/books.csv'
        books = pd.read_csv(file_path)
        return books
    
    except FileNotFoundError as e:
        print(f"Error: The file 'books.csv' was not found at path {file_path}.")
        raise  # raise exception if file not found


def basic_preprocessing(books):
    """Perform basic preprocessing on the book data."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        books["missing_description"] = np.where(books["description"].isna(),1,0)
        books["age_of_book"] = CURRENT_YEAR - books["published_year"]
        return books
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise e
    

def remove_null_observations(books):
    """Remove observations with null values in critical fields."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        books = books[
            ~(books['description'].isna())&
            ~(books['num_pages'].isna())&
            ~(books['average_rating'].isna())&
            ~(books['published_year'].isna())
        ]
        return books
    
    except Exception as e:
        logger.error(f"An error occurred during null removal: {e}")
        raise e


def cut_off_description_length(books, min_length=25):
    """Remove books with description length below a certain threshold."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        books['description_word_count'] = books['description'].str.split().str.len()
        books = books[books['description_word_count'] >= min_length]
        return books
    except Exception as e:
        logger.error(f"An error occurred during description length cutoff: {e}")
        raise e
    


def aggregate_title_and_subtitle(books):
    """Aggregate book titles and subtitles into a single field."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed
        books['title_and_subtitle'] = np.where(books['subtitle'].isna(),
                                    books['title'],
                                    books['title'].str.cat(books['subtitle'], sep = ': ')
                                          )

        return books
    
    except Exception as e:
        logger.error(f"An error occurred during aggregation: {e}")
        raise e  


def tag_description(books):
    """Tag book descriptions with special markers."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        books["tagged_description"] = books['isbn13'].astype(str).str.cat(books['description'], sep=' ')
        return books
    except Exception as e:
        logger.error(f"An error occurred during description tagging: {e}")
        raise e


def map_category(books):
    """Map book categories to broader genres."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        category_mapping = {'Fiction' : "Fiction",
        'Juvenile Fiction': "Children's Fiction",
        'Biography & Autobiography': "Nonfiction",
        'History': "Nonfiction",
        'Literary Criticism': "Nonfiction",
        'Philosophy': "Nonfiction",
        'Religion': "Nonfiction",
        'Comics & Graphic Novels': "Fiction",
        'Drama': "Fiction",
        'Juvenile Nonfiction': "Children's Nonfiction",
        'Science': "Nonfiction",
        'Poetry': "Fiction"}

        # categories besides these are mapped to NaN
        books["simple_categories"] = books['categories'].map(category_mapping) 
        return books
    except Exception as e:
        logger.error(f"An error occurred during category mapping: {e}")
        raise e
    

def init_zero_shot_classifier():    
    """Classify book descriptions into genres using zero-shot classification."""
    try:
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    except Exception as e:
        logger.error(f"An error occurred while fetching HUGGINGFACEHUB_API_TOKEN: {e}")
        raise e
    
    try:
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              use_auth_token=HUGGINGFACEHUB_API_TOKEN)

        categories = ["Fiction","Nonfiction"]
        return classifier, categories

        
    except Exception as e:
        logger.error(f"An error occurred during intilialzing zero-shot classifier: {e}")
        raise e
    

def generate_predictions(sequence, classifier, categories):
    """Generate zero-shot classification predictions for a given sequence."""
    prediction = classifier(sequence, categories)
    # print(prediction)
    max_index = np.argmax(prediction["scores"])
    max_label = prediction["labels"][max_index]
    return max_label


def classify_missing_categories(books, classifier, categories):
    """Classify books with missing categories using zero-shot classification."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        isbns = []
        predicted_cats = []

        missing_cats = books.loc[books['simple_categories'].isna(), ["isbn13", "description"]].reset_index(drop=True)

        for i in range(len(missing_cats.index)):
            sequence = missing_cats['description'][i]
            predicted_cats += [generate_predictions(sequence, classifier, categories)]
            isbns += [missing_cats["isbn13"][i]]
        
        missing_predicted_df = pd.DataFrame({"isbn13":isbns, "predicted_categories":predicted_cats})
        return missing_predicted_df

    except Exception as e:
        logger.error(f"An error occurred during classification of missing categories: {e}")
        raise e
    

def merge_books_and_classified_data(books, missing_predicted_df):
    """Merge original book data with classified missing categories."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        books = pd.merge(books, missing_predicted_df, on = 'isbn13', how = 'left')
        books['simple_categories'] = np.where(books['simple_categories'].isna(),
                                      books["predicted_categories"],
                                      books["simple_categories"])
        books = books.drop(columns = ["predicted_categories"])
        return books

    except Exception as e:
        logger.error(f"An error occurred during merging of classified data: {e}")
        raise e
    
def save_processed_data(books):
    """Save the processed book data to a CSV file."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        output_path = 'data/interim/books_with_categories.csv'
        books.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving processed data: {e}")
        raise e
    

def main():
    """Main function to orchestrate data ingestion and preprocessing."""
    try:
        books = ingest_data()
        books = basic_preprocessing(books)
        books = remove_null_observations(books)
        books = cut_off_description_length(books)
        books = aggregate_title_and_subtitle(books)
        books = tag_description(books)
        books = map_category(books)
        
        classifier, categories = init_zero_shot_classifier()
        missing_predicted_df = classify_missing_categories(books, classifier, categories)
        books = merge_books_and_classified_data(books, missing_predicted_df)
        
        save_processed_data(books)
        
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")
        raise e
    

if __name__ == "__main__":
    main()