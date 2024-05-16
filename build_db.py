import chromadb
import pandas as pd

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document


def load_documents(filename):
    """
    This function loads preprocessed documents for the vectorstore based on an input csv file.
    TODO: fix this
    """
    gloger_df = pd.read_csv(filename)

    text_splitter = TokenTextSplitter(
        # For all-MiniLM-L6-v2
        # chunk_size=450, # 500 tokens is the max
        # chunk_overlap=100 # Overlap of N tokens between chunks (to reduce chance of cutting out relevant connected text like middle of sentence)
        # For BAAI/bge-m3
        chunk_size=1024,
        chunk_overlap=128
    )

    documents = []

    for index, row in gloger_df[gloger_df['Content'].notnull()].iterrows():
        chunks = text_splitter.split_text(row.Content)
        total_chunks = len(chunks)
        for chunk_num in range(1,total_chunks+1):
            header = f"Quelle: {row['Source']}\(Textauszug {chunk_num} von {total_chunks})\n\n"
            
            chunk = chunks[chunk_num-1]
            documents.append(Document(page_content=header + chunk, metadata={"source": "local"}))

    return documents


if __name__ == "__main__":
    author_name = "gloger_gotthold"

    # Load Embedding model
    # all-MiniLM-L6-v2  # 384dim
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # BAAI/bge-m3  # 1024dim
    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-m3")

    # Load documents
    documents = load_documents(f"{author_name}.csv")

    # Create chromadb
    client = chromadb.PersistentClient(path="data")

    collection = client.create_collection(
        name=author_name, 
        embedding_function=embedding_function
        )
    
    collection.add(documents)
    print("Done?")


# collection = client.get_collection(name="gloger_gotthold", embedding_function=embedding_function)