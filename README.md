# GDR Authors RAG

This work in progress repository contains a `Streamlit` application that allows users to ask questions about authors of the former GDR / Eastern Germany.

The first version (currently on `main` branch) uses a pre-trained model via [Groq](https://groq.com/)'s API to find relevant excerpts from lexicon articles and generate responses to the user's questions.

The second version (currently on `dev` branch) takes text as input and extracts relevant information based on a set of pre-defined prompts to return structured output.

## Features

### Version 1 - RAG

- **Question-Answering System**: Users can ask questions about authors, and the application will generate responses based on relevant excerpts from lexicon articles.

- **Customization**: Users can provide additional context for the model and choose from a list of pre-trained models.

- **Similarity Search**: The application uses a vector store to find the most relevant excerpts from lexicon articles based on the user's question.

### Version 2 - Information Extraction

- **Information Extraction**: Users can paste lexicon articles and choose from a set of pre-defined prompts to extract relevant information.
