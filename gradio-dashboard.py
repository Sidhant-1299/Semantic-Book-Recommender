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

embeddings = OpenAIEmbeddings(api_key = os.getenv('OPENAI_API_KEY',None))
# vector_store_path = Chroma(
#     embedding_function=embeddings,
#     persist_directory="./chroma_book_db"
# )

# db_books = Chroma.from_documents(documents)

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
        authors_split = row['authors'].split(";")
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

def show_book_details(evt: gr.SelectData):
    book = last_recommendations[evt.index]  # use the stored global data
    html = f"""
    <div class="modal">
        <div class="modal-content">
            <img src="{book['image']}" style="width:100%; border-radius: 10px; margin-bottom: 15px;">
            <h2>{book['label']}</h2>
            <p>{book['description']}</p>
        </div>
    </div>
    """
    return gr.update(value=html, visible=True)

# Revolutionary CSS for full-width Netflix-style dashboard
custom_css = """
body { background-color: #111; color: #fff; font-family: 'Helvetica', sans-serif; }

/* Header */
.header { text-align: center; margin-bottom: 40px; }
.header h1 { font-size: 3rem; margin: 0; }
.header p { color: #ccc; font-size: 1.2rem; }

/* Inputs */
.inputs { display: flex; gap: 15px; justify-content: center; margin-bottom: 40px; }
.gr-textbox, .gr-dropdown { border-radius: 6px !important; padding: 10px; background-color: #222 !important; color: #fff; }

/* Button */
.custom-btn {
    background: #e50914 !important;
    border-radius: 6px;
    padding: 12px 24px;
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s;
}
.custom-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.6);
}

/* Gallery as horizontal scroll row */
.gr-gallery { display: flex; overflow-x: auto; gap: 20px; padding-bottom: 20px; scroll-behavior: smooth; }
.gr-gallery img { border-radius: 8px; height: 300px; transition: transform 0.2s, filter 0.2s; cursor: pointer; }
.gr-gallery img:hover { transform: scale(1.1); filter: brightness(1.2); }

/* Modal for book details */
.modal {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: #222;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.8);
    max-width: 600px;
    z-index: 9999;
}
.modal h2 { margin-top: 0; color: #fff; }
.modal p { color: #ccc; line-height: 1.5; }

/* Close modal button */
.modal::after {
    content: '✖';
    position: absolute;
    top: 10px; right: 15px;
    cursor: pointer;
    font-size: 1.2rem;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as dashboard:
    gr.HTML(
        """
        <div class="header">
            <h1>Semantic Book Recommender</h1>
            <p>Search smarter, Netflix-style</p>
        </div>
        """
    )

    with gr.Row(elem_classes="inputs"):
        user_query = gr.Textbox(placeholder="A thrilling mystery in Victorian London...", lines=1)
        category_dropdown = gr.Dropdown(choices=categories, value="All")
        tone_dropdown = gr.Dropdown(choices=tone, value="All")
        recommend_button = gr.Button("Recommend", elem_classes="custom-btn")

    gr.Markdown("<h2 style='margin-left:20px;'>Recommended Books</h2>")
    gallery = gr.Gallery(columns=4, rows=1, object_fit="cover", height="auto", allow_preview=False)
    book_detail_box = gr.HTML(visible=False)

    recommend_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

    gallery.select(
        fn=show_book_details,
        inputs=[],
        outputs=book_detail_box
    )

dashboard.launch()