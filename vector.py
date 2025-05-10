from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# pandas dataframe
# Load the CSV file into a pandas DataFrame
df=pd.read_csv("realistic_restaurant_reviews.csv")

# emberddings
# Create a list of Document objects from the DataFrame
embeddings= OllamaEmbeddings(model="mxbai-embed-large") #model="mxbai-embed-large" is the default
db_location = "./chrome_langchain_db"

# Check if the database directory exists
add_documents= not os.path.exists(db_location)

# If the directory does not exist, create it
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document=Document(
            page_content=row["Title"] + " " + row["review"],
            metadata={"rating": row["rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create a Chroma vector store
vectore_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add the documents to the vector store
if add_documents:
    vectore_store.add_documents(documents=documents, ids=ids)

# connecting to the vector store and llm
retriever = vectore_store.as_retriever(search_kwargs={"k": 5})