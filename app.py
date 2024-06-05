import streamlit as st

from groq import Groq
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def category_prompts(selected_category):
    # TODO
    # one-shot prompt Helga Novak  
    prompts = {
        "Ausbildung": "Extrahiere Informationen über Ausbildung.",
        "Beruf": "Extrahiere Informationen über den Beruf.",
        "Mitgliedschaften": "Extrahiere Informationen über Mitgliedschaften.",
        "Lebensorte": "Extrahiere Informationen über Lebensorte."
        # Add other categories as needed
    }

    return prompts[selected_category]


def information_extraction(client, model, user_question, selected_category):
    """
    Extracts specific information from the user provided text based on selected categories.
    """
    base_prompt = '''
    Du extrahierst Informationen aus einem Text und gibst diese in strukturierter Form zurück.
    Basierend auf einem Text den der Nutzer zur Verfügung stellt, extrahierst du die relevanten
    Informationen und gibst sie in strukturierter Form zurück. Füge nur die Informationen hinzu, 
    die eindeutig aus dem Text extrahierbar sind. Fehlen Informationen, dann lasse die Felder 
    entsprechend leer.
    '''
    
    system_prompt = base_prompt + f"\n- {category_prompts(selected_category)}"

    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":  system_prompt
            },
            {
                "role": "user",
                "content": user_question
            }
        ],
        model = model,
        temperature = 0,
        # seed = seed  # For reproducibility?
    )
    
    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response


def main():
    """
    This is the main function that runs the application. It initializes the Groq client,
    gets user input from the Streamlit interface,  generates a response to the user's 
    question using a pre-trained model, and displays the response.
    """
    # Initialize the Groq client
    groq_api_key = os.environ.get('GROQ_API_KEY')

    client = Groq(
        api_key=groq_api_key
    )

    # Display the title and introduction of the application
    st.title("Structured Information Extraction")
    multiline_text = """
    Wähle einen Ereignistyp aus, zu dem du Attribute aus einem Lexikonartikel extrahieren möchtest:
    """

    # TODO Add more categories as needed
    # Radio button options for categories
    categories = ["Ausbildung", "Beruf", "Mitgliedschaften", "Lebensorte"]  
    selected_category = st.radio("Ereignistyp auswählen:", categories)

    st.markdown(multiline_text, unsafe_allow_html=True)
   
    # TODO add option to change model?
    model = 'mixtral-8x7b-32768'

    # Get the user's question
    user_question = st.text_input(
        "Füge den Lexikonartikel ein, aus dem Informationen extrahieren möchtest:",
        placeholder="Geben Sie hier Ihren Text ein...")

    if user_question:
        response = information_extraction(client, model, user_question, selected_category)
        st.write(response)


if __name__ == "__main__":
    main()
