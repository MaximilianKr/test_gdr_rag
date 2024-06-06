# GDR Authors RAG / Structured Information Extraction

This work in progress repository contains a `Streamlit` application that allows users to ask questions about authors of the former GDR / Eastern Germany.

The first version (currently on `main` branch) uses a pre-trained model via [Groq](https://groq.com/)'s API to find relevant excerpts from lexicon articles and generate responses to the user's questions.

The second version (currently on `dev` branch) takes text as input and extracts relevant information based on a set of pre-defined prompts to return structured output.

## Overview

- [GDR Authors RAG / Structured Information Extraction](#gdr-authors-rag--structured-information-extraction)
  - [Overview](#overview)
  - [Features](#features)
    - [Version 1 - RAG](#version-1---rag)
    - [Version 2 - Information Extraction](#version-2---information-extraction)
  - [Setup](#setup)
    - [venv](#venv)
    - [conda](#conda)
  - [Usage](#usage)
  - [ToDo](#todo)
  - [Author](#author)

## Features

### Version 1 - RAG

- **Question-Answering System**: Users can ask questions about authors, and the application will generate responses based on relevant excerpts from lexicon articles.

- **Customization**: Users can provide additional context for the model and choose from a list of pre-trained models.

- **Similarity Search**: The application uses a vector store to find the most relevant excerpts from lexicon articles based on the user's question.

### Version 2 - Information Extraction

- **Information Extraction**: Users can paste lexicon articles and choose from a set of pre-defined prompts to extract relevant information.

## Setup

### venv

- recommended: use [uv package manager](https://github.com/astral-sh/uv) for a fast setup

```shell
uv venv
```

```shell
# macOS / Linux
source .venv/bin/activate
```

```shell
# Windows
.venv\Scripts\activate
```

```shell
uv pip install -r requirements.txt
```

### conda

```shell
conda env create -f environment.yml
```

```shell
conda activate gdr
```

## Usage

```shell
streamlit run app.py
```

- Select one of the provided models from Groq or Google.
- Choose a prompt template for information extraction.
- Paste your text / lexicon article and press `Ctrl + Enter` and wait for the reply.

## ToDo

- [ ] check prompt templates
- [ ] separate RAG / information extraction apps
  - [ ] merge branches
- [ ] test coverage with `pytest`

## Author

- Maximilian Krupop

[Back to Top](#gdr-authors-rag--structured-information-extraction)
