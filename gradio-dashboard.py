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


categories = ["All"] + sorted(books["simple_categories"].unique())
tone = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown(
        """
        <h1 style="text-align:center; margin-bottom: 0.5rem;">Semantic Book Recommender</h1>
        <p style="text-align:center; opacity: 0.7;">Search smarter, get deeper recommendations</p>
        """
    )

    with gr.Group():
        with gr.Column():
            user_query = gr.Textbox(
                label="Book title or description",
                placeholder="e.g., A thrilling mystery in Victorian London...",
                lines=2
            )

            with gr.Row():
                category_dropdown = gr.Dropdown(
                    label="Category",
                    choices=categories,
                    value="All"
                )
                tone_dropdown = gr.Dropdown(
                    label="Tone",
                    choices=tone,
                    value="All"
                )
                recommend_button = gr.Button(
                    "Recommend",
                    variant="primary"
                )

    gr.Markdown("<h2 style='margin-top: 2rem;'>Recommended Books</h2>")

    gallery = gr.Gallery(
        label="Book Recommendations",
        columns=4,
        rows=2,
        object_fit="contain"
    )

    recommend_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

if __name__ == "__main__":
    dashboard.launch()
