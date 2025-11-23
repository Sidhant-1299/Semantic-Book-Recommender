import gradio as gr
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"]+ "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isnull(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1, chunk_overlap = 0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents,
                                OpenAIEmbeddings(api_key = os.getenv('OPENAI_API_KEY',None)),
                                persist_directory="chroma_book_db")


def retrieve_semantic_recommendations(
            query: str,
            category: str = None, 
            tone: str = None, 
            intial_top_k: int = 50, 
            final_top_k: int = 16 
        ) -> pd.DataFrame:

    recs = db_books.similarity_search(query=query, k = intial_top_k)
    #get isbns of similar books
    books_list = [int(rec.page_content.split()[0].strip().replace(",","").replace('"','')) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(intial_top_k)

    if category != "All":   
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    
    #lookup table
    mapping = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness"
    }

    col = mapping.get(tone)
    if col is None:
        return book_recs #default
    
    return book_recs.sort_values(by=col, ascending=False)



def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row['authors'].split(";")
        if (authors_split_len:= len(authors_split)) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif (authors_split_len) > 2:
            authors_str = f"{authors_split[0]}{", ".join(authors_split[1:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categoris = ["All"] + sorted(books["simple_categories"].unique())
tone = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Enter a book title or description",
                                 placeholder = "e.g., A thrilling mystery novel set in Victorian London...")
        category_dropdown = gr.Dropdown(label = "Select Category",
                                        choices = categoris,
                                        value = "All")
        tone_dropdown = gr.Dropdown(label = "Select Tone",
                                    choices = tone,
                                    value = "All")
        recommend_button = gr.Button("Get Recommendations")
    
    gr.Markdown("## Recommended Books:")
    gallery = gr.Gallery(label = "Book Recommendations",
                         columns = 8,
                         rows=2)
    
    recommend_button.click(fn=recommend_books,
                           inputs=[user_query, category_dropdown, tone_dropdown],
                           outputs=gallery)
    
if __name__ == "__main__":
    dashboard.launch()
