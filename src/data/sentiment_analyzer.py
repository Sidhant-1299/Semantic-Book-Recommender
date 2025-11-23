import numpy as np
import pandas as pd
import logging
from transformers import pipeline,AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os
import dotenv
from safetensors.torch import save_file
import torch

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='sentiment_analyzer.log',
                    filemode='a')

emotional_labels = ["anger","disgust","fear","joy","neutral", "sadness","suprise"]
isbn = []
emotions_scores = {label: [] for label in emotional_labels}


def load_model():
    """
    Load the model using safetensors
    """

    try:
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        model = AutoModelForSequenceClassification.from_pretrained(
                      model_name,
                      use_safetensors=True,)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None
        )
        return classifier
    

    except Exception as e:
        logger.error(f"An error occurred during sentiment classification: {e}")
        raise e
    


def calculate_max_emotion_scores(predictions):
  per_emotion_scores = {label:[] for label in emotional_labels}
  for prediction in predictions:
    sorted_prediction = sorted(prediction, key = lambda x: x['label'])
    for index, label in enumerate(emotional_labels):
      per_emotion_scores[label].append(sorted_prediction[index]["score"])
  return {label: np.max(scores) for label, scores in per_emotion_scores.items()}


def predict_sentiments(books, classifier):
    """Predict sentiments for book descriptions using the provided model."""
    try:
        if books.empty:
            raise ValueError("Books dataset empty")  # Return empty DataFrame if ingestion failed

        for i in tqdm(range(len(books))):
            isbn.append(books["isbn13"][i])
            sentences = books["description"][i].split(".")
            predictions = classifier(sentences)
            max_scores = calculate_max_emotion_scores(predictions)
            for label in emotional_labels:
                emotions_scores[label].append(max_scores[label])


        emotions_df = pd.DataFrame(emotions_scores)
        emotions_df["isbn13"] = isbn
        return emotions_df

    except Exception as e:
        logger.error(f"An error occurred during sentiment prediction: {e}")
        raise e
    

def merge_sentiments(books, emotions_df):
    """Merge the sentiment scores with the original books DataFrame."""
    try:
        merged_df = books.merge(emotions_df, on="isbn13", how="left")
        return merged_df

    except Exception as e:
        logger.error(f"An error occurred during merging sentiments: {e}")
        raise e
    

def main():
    try:
        file_path = 'data/interim/books_with_categories.csv'
        books = pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"Error: The file was not found at path {file_path}.")
        raise  # raise exception if file not found

    # save_model_as_safetensors()
    classifier = load_model()
    emotions_df = predict_sentiments(books, classifier)
    merged_df = merge_sentiments(books, emotions_df)
    output_path = 'data/final/books_with_emotions.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Sentiment analysis completed and saved to {output_path}")


if __name__ == "__main__":
    main()
