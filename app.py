import streamlit as st

import pandas as pd
import numpy as np
from groq import Groq
import os

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def get_relevant_excerpts(user_question, docsearch):
    """
    This function retrieves the most relevant excerpts from lexicon articles based on the user's question.

    Parameters:
    user_question (str): The question asked by the user.
    docsearch (ChromaDBVectorstore): The Pinecone vector store containing the lexicon articles.

    Returns:
    str: A string containing the most relevant excerpts from lexicon articles.
    """

    # Perform a similarity search on the Pinecone vector store using the user's question
    relevent_docs = docsearch.similarity_search(user_question)

    # Extract the page content from the top 3 most relevant documents and join them into a single string
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevent_docs[:3]])

    return relevant_excerpts


def rag_chat_completion(client, model, user_question, relevant_excerpts, seed=42):
    """
    This function generates a response to the user's question using a pre-trained model.

    Parameters:
    client (Groq): The Groq client used to interact with the pre-trained model.
    model (str): The name of the pre-trained model.
    user_question (str): The question asked by the user.
    relevant_excerpts (str): A string containing the most relevant excerpts from lexicon articles.

    Returns:
    str: A string containing the response to the user's question.
    """

    # Define the system prompt
    # system_prompt = '''
    # Du bist ein Historiker, der sich auf Autoren der ehemaligen DDR spezialisiert hat. 
    # Beantworte die Frage des Nutzers auf Grundlage der entsprechenden Textauszüge aus DDR-Lexikonartikeln. 
    # Gib die Quelle mit an aus der die Informationen verwendest (verweise nicht auf den Textauszug).
    # Wenn keiner der Textauszüge relevante Informationen erhält, dann teile mit, dass du die Frage anhand
    # der Daten nicht beantworten kannst.
    # '''

    system_prompt = '''
    Du bist ein Historiker, der sich auf Autoren der ehemaligen DDR spezialisiert hat. 
    Beantworte die Frage des Nutzers auf Grundlage der entsprechenden Textauszüge aus DDR-Lexikonartikeln. 
    Gib den Namen der Quelle an, aus der die Informationen stammen (ohne den Textauszug). 
    Wenn keiner der Textauszüge relevante Informationen enthält, teile dem Nutzer mit, 
    dass du die Frage anhand der verfügbaren Daten nicht beantworten kannst.
    '''

    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":  system_prompt
            },
            {
                "role": "user",
                "content": "Nutzerfrage: " + user_question + "\n\nRelevante Textauszüge:\n\n" + relevant_excerpts,
            }
        ],
        model = model,
        temperature = 0,
        # seed = seed  # For reproducibility?
    )
    
    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response


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
    

def main():
    """
    This is the main function that runs the application. It initializes the Groq client and the SentenceTransformer model,
    gets user input from the Streamlit interface, retrieves relevant excerpts from lexicon articles based on the user's question,
    generates a response to the user's question using a pre-trained model, and displays the response.
    """

    # Load Embedding model
    
    # all-MiniLM-L6-v2  # 384dim
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # BAAI/bge-m3  # 1024dim
    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-m3")
    
    # Initialize the Groq client
    groq_api_key = os.environ.get('GROQ_API_KEY')

    # groq_api_key = st.secrets["GROQ_API_KEY"]
    # pinecone_api_key=st.secrets["PINECONE_API_KEY"]
    # pinecone_index_name = "presidential-speeches"
    client = Groq(
        api_key=groq_api_key
    )

    # Load documents
    documents = load_documents("gloger_gotthold.csv")

    # Initialize vector database
    # TODO: load chroma from disk or use pinecone
    # pc = Pinecone(api_key = pinecone_api_key)
    # docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
    docsearch = Chroma.from_documents(documents, embedding_function)

    # Display the title and introduction of the application
    st.title("DDR RAG Test")
    multiline_text = """
    Stelle Fragen zu Autoren der ehemaligen DDR. Die App ordnet der Frage relevante Auszüge aus Lexikonartikeln zu und generiert eine Antwort mithilfe eines vortrainierten Modells. Zum Beispiel:\n
    ```
    Wann wurde Gloger Gotthold geboren?
    ```\n
    ```
    Wo studierte Gloger Gotthold?
    ```\n
    ```
    Welche Werke veröffentlichte Gloger Gotthold?
    ```
    """

    st.markdown(multiline_text, unsafe_allow_html=True)
   
    model = 'mixtral-8x7b-32768'

    # Get the user's question
    user_question = st.text_input("Gib deine Frage ein:")

    if user_question:
        relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
        response = rag_chat_completion(client, model, user_question, relevant_excerpts)
        st.write(response)


if __name__ == "__main__":
    main()

# TODO:

# - test chunk length / overlap iterations --> depends on embedding model!
# - use persistent chroma client
# - fix load_documents(filename)
# - rewrite README.md
# - rewrite requirements / environment
# - add pinecone?
# - check system prompt for consistency / iterate
# - test other embedding function
# - add streaming from groq / streamlit
# - add more authors --> append author name to every chunk to not mix up results?