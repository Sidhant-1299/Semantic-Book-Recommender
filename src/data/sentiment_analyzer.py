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


def save_model_as_safetensors():
    model_id = "j-hartmann/emotion-english-distilroberta-base"
    local_dir = "models/emotion_model"

    # Download the model and save it to the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Save the model using the safe_serialization=True argument
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir, safe_serialization=True)
    print(f"Model saved locally to {local_dir}. It should contain 'model.safetensors'.")


def load_model():
    """
    Load the model from the local directory, enforcing safetensors loading 
    and explicit CPU usage for Intel Mac.
    """
    model_dir = "models/emotion_model"  # Must contain model.safetensors + tokenizer
    device_to_use = "cpu"

    try:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model folder {model_dir} not found. "
                "Please run Step A to download and convert the model."
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model weights from safetensors
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,   # Ensure we only use local files
            safe_serialization=True  # Force loading of safetensors
        )

        # Load pipeline and pass the CPU device
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_to_use, # <-- Set to 'cpu' for Intel Mac
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

    save_model_as_safetensors()
    classifier = load_model()
    emotions_df = predict_sentiments(books, classifier)
    merged_df = merge_sentiments(books, emotions_df)
    output_path = 'data/final/books_with_emotions.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Sentiment analysis completed and saved to {output_path}")


if __name__ == "__main__":
    main()
