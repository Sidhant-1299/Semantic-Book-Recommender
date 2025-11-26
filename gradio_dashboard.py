import gradio as gr
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from gradio_modal import Modal

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY',None)
if not api_key:
    raise ValueError("OPENAI_API_KEY missing. Please add it in your env file")

persisent_vector_dir = "./chroma_book_db"

data = "data/final/books_with_emotions.csv"
books = pd.read_csv(data)

books["large_thumbnail"] = books["thumbnail"]+ "&fife=w800"
default_cover = "src/assets/cover-not-found.jpg"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isnull(), default_cover, books["large_thumbnail"])

document = "data/final/tagged_description.txt"
raw_documents = TextLoader(document).load()
text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1, chunk_overlap = 0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(api_key)


#check if persistent directory exists
if not os.path.isdir(persisent_vector_dir):
    os.mkdir(persisent_vector_dir)

# make chroma db if persisent vector dir is empty and perist it
if not os.listdir(persisent_vector_dir):
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding =embeddings, 
        persist_directory=persisent_vector_dir
    )
# if it is not empty use it
else:
    vectordb = Chroma(
        persist_directory=persisent_vector_dir,
        embedding_function = embeddings
    )


def retrieve_semantic_recommendations(
            query: str,
            category: str = None, 
            tone: str = None, 
            intial_top_k: int = 50, 
            final_top_k: int = 24 
        ) -> pd.DataFrame:

    recs = vectordb.similarity_search(query=query, k = intial_top_k)
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


categories = ["All"] + sorted(books["simple_categories"].unique())
tone = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Global variable to store the current recommendation batch
last_recommendations = []

def recommend_books(query, category, tone):
    global last_recommendations
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    last_recommendations = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_description = " ".join(description.split()[:30]) + "..."
        # empty string if authors in NaN
        authors_raw = row["authors"] if isinstance(row["authors"], str) else ""
        authors_split = authors_raw.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{authors_split[0]}, {', '.join(authors_split[1:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        last_recommendations.append({
            "image": row["large_thumbnail"],
            "label": row["title"],
            "description": caption
        })

        results.append(row["large_thumbnail"])  # Gallery only needs images

    return results


def load_css(filename:str = 'dashboard.css'):
    with open(filename, 'r') as rf:
        css_content = rf.read()
    return css_content


def show_book_details(evt: gr.SelectData):
    global last_recommendations
    
    book = last_recommendations[evt.index]
    book_img_html = f'<div class="modal-image-container"><img src="{book["image"]}" /></div>'
    
    return (
        book_img_html,
        f"<h2>{book['label']}</h2>",
        f"<p>{book['description']}</p>",
        gr.update(visible=True, elem_classes="modal-active") # This command now goes to the MODAL
    )

with gr.Blocks(theme=gr.themes.Base(), css=load_css()) as dashboard:
    gr.HTML(
        """
        <div class="header">
            <h1>Semantic Book Recommender</h1>
            <p>Search smarter</p>
        </div>
        """
    )

    with gr.Row(elem_classes="inputs"):
        user_query = gr.Textbox(
            label = "Please enter a description of the book",
            placeholder="A thrilling mystery in Victorian London...", lines=1,)
        category_dropdown = gr.Dropdown(choices=categories, value="All", label = "Select a category")
        tone_dropdown = gr.Dropdown(choices=tone, value="All", label = "Select an emotional tone")
        recommend_button = gr.Button("Recommend", elem_classes="custom-btn")

    gr.Markdown("<h2 style='margin-left:20px;'>Recommended Books</h2>")
    gallery = gr.Gallery(
        columns=4,
        rows=1,
        object_fit="cover",
        height="auto",
        allow_preview=False
    )

    with Modal(visible=False) as book_modal:
        # We wrap the content in a generic Group/Column to act as our "Card"
        with gr.Column(elem_classes="modal-body"):
            book_img = gr.HTML("") 
            book_title = gr.HTML("")
            book_desc = gr.HTML("")

        

    # Recommend button updates gallery
    recommend_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

    gallery.select(
        fn=show_book_details,
        inputs=None,
        outputs=[book_img, book_title, book_desc, book_modal] 
    )


if __name__ == "__main__":    
    dashboard.launch()