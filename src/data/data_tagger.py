import pandas as pd
import logging

logger  = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_tagger.log',
                    filemode='a')

def generate_tagged_description_file(file_path):
    """Generate a text file containing tagged descriptions from the books DataFrame."""
    try:
        books = pd.read_csv(file_path)
        tagged_description_file_path = 'data/final/tagged_description.txt'
        # print(books.head())
    except FileNotFoundError as e:
        logger.error(f"Error: The file was not found at path {file_path}.")
        raise  # raise exception if file not found
    try:
        # print(books['tagged_description'][0])
        books['tagged_description'].to_csv(
            tagged_description_file_path,
            index=False,
            header=False,
            sep='\n'
        )
    except Exception as e:
        logger.error(f"An error occurred while saving tagged description to file {tagged_description_file_path}\n{e}")
        raise e


def main():
    file_path = 'data/interim/books_with_categories.csv'
    generate_tagged_description_file(file_path)

if __name__ == "__main__":
    main()