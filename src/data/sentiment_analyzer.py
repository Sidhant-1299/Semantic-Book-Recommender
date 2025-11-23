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


# def save_model_as_safetensors():
#     """
#     Downloads the model and saves it locally, explicitly forcing the use of 
#     safetensors for secure local loading.
#     """
#     model_id = "j-hartmann/emotion-english-distilroberta-base"
#     model_dir = "models/emotion_model"

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # 1. Load the model from the Hugging Face Hub.
#     # *** FIX: Removed the problematic 'safe_serialization=True' argument from here. ***
#     model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
#     # 2. Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     # 3. Save the model locally, explicitly forcing the safetensors format.
#     # This argument *belongs* here, in the save function.
#     model.save_pretrained(model_dir, safe_serialization=True)
#     tokenizer.save_pretrained(model_dir)
    
#     # 4. Save the config
#     model.config.save_pretrained(model_dir)
#     logger.info(f"Model successfully downloaded and saved as safetensors in {model_dir}")



def load_model():
    """
    Load the model from the local directory, enforcing safetensors loading 
    and explicit CPU usage for Intel Mac.
    """
    # model_dir = "models/emotion_model"  # Must contain model.safetensors + tokenizer
    # device_to_use = "cpu"

    # try:
    #     if not os.path.exists(model_dir):
    #         raise FileNotFoundError(
    #             f"Model folder {model_dir} not found. "
    #             "Please run Step A to download and convert the model."
    #         )

    #     # Load tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
    #     # Load model weights from safetensors
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         model_dir,
    #         local_files_only=True,   # Ensure we only use local files
    #         safe_serialization=True  # Force loading of safetensors
    #     )

    #     # Load pipeline and pass the CPU device
    #     classifier = pipeline(
    #         "text-classification",
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device_to_use, # <-- Set to 'cpu' for Intel Mac
    #         top_k=None
    #     )

    #     return classifier

    try:
        classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None,
                      use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                      use_safetensors=True,)
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
