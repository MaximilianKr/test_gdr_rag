import os
import streamlit as st

from groq import Groq
import google.generativeai as genai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def category_prompts(selected_category):
    # TODO
    # one-shot prompt Helga Novak
    prompts = {
        "Lebensdaten, Familie und persönliches Netzwerk": 
            """
            Geburt: Individualereignis
            
            Tod: Individualereignis
            
            Staatsbürgerschaft: Abstrakte Mitgliedschaft
            
            Bekenntnis: Individualereignis
            - Religion/Weltanschauung/Konfession
            - Art des Eintritts
            
            Herkunft: Individualereignis
            - soziale Klasse, Konfession, Herkunft von Familienangehörigen
            
            Elternschaft: Individualereignis
            - Geschlechtsidentität
            - Name des Kindes
            - Art der Beendigung
            
            Verwaisung: Individualereignis
            
            Bekanntschaft/Beziehung: Individualereignis
            - Art der Bekanntschaft (d.h. Freundschaft, Partnerschaft, Ehe etc. mit Zeitangaben)
            - Angaben zur Bezugsperson (soziale Klasse, Konfession, Herkunft etc.)
            
            Ereignisteilnahme: Individualereignis
            - Ereignis
            - Rollen
            """,
        "Beruf und Ausbildung":
            """
            Ausbildung: Abstrakte Mitgliedschaft
	        - Ausbildungsart
	        - Ausbildungsinhalte
	        - Abschluss

            Berufstätigkeit (inkl. literarische Tätigkeit): Abstrakte Mitgliedschaft
	        - Berufsbezeichnung

            Lehrtätigkeit:
	        - Lehrgebiete

            Militär/Kriegsdienst (inkl. Gefangenschaft):
	        - Art des Diensteintritts
	        - Kriegsteilnahme

            Beschäftigungslosigkeit: Individualereignis
	        - Art der Beschäftigungslosigkeit (z.B. Arbeitslosigkeit, Rente etc.)
            """,
        "Mitgliedschaften": "Mitgliedschaft: Abstrakte Mitgliedschaft",
        "Lebensorte": 
            """
            Wohnung/Aufenthalt: Individualereignis
            - Art des Aufenthalts
            - Migrationsziel?
            - Ausgangsort (bei Migration)
            - Migrationstyp
            """,
        "Würdigung": 
            """
            Würdigung: Individualereignis
            - Würde/Preis
            - Datum
            - Status
            - Würdigungstyp
            - Institution
            - Art der Beendigung

            Zuwendung: Individualereignis
            - Zuwendungsgeber
            - Zuwendungstyp
            - Status (z.B. beantragt, gewährt, abgelehnt)
            """,
        "Sonstiges": 
            """
            Zwangsmaßnahme/Verfolgung: Individualereignis
            - Institution
            - Art der Maßnahme
            - Begründung
            - Beendigungsart

            Benanntes Ereignis: Ereignis
            - Bezeichnungen
            """,
    }

    return prompts[selected_category]


def groq_response(model, system_prompt, user_question):
    # Initialize the Groq client
    groq_api_key = os.environ.get('GROQ_API_KEY')

    client = Groq(
        api_key=groq_api_key
    )

    if "Mixtral" in model:
        model = 'mixtral-8x7b-32768'
    elif "3-8B" in model:
        model = 'llama3-8b-8192'
    elif "3-70B" in model:
        model = 'llama3-70b-8192'

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


def google_response(model, system_prompt, user_question):
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=google_api_key)

    if "Flash" in model:
        model = genai.GenerativeModel('gemini-1.5-flash')
    
    response = model.generate_content(system_prompt+ "\n" + user_question).text

    return response
                    


def information_extraction(client, model, user_question, selected_category):
    """
    Extracts specific information from the user provided text based on selected categories.
    """
    base_prompt = f"""
    Basierend auf einem Text den der Nutzer zur Verfügung stellt, extrahierst du die relevanten
    Informationen und gibst sie in strukturierter Form zurück. Füge nur die Informationen hinzu, 
    die eindeutig aus dem Text extrahierbar sind. Sind Informationen zu einem Ereignis unbekannt 
    oder fehlen, dann füge das Feld entsprechend nicht hinzu.

    Es gibt folgende Ereignistypen mit den jeweiligen Attributen:
    
    Ereignis
    - Beginn
    - Ende
    - Dauer
    - Ort
    - Beteiligte (mit Angabe von Rolle und ggf. Zeitangaben)
    - Vorhergehende Ereignisse
    - Zeitgleiche Ereignisse

    Individualereignis: Ereignis
    - Protagonist

    Abstrakte Mitgliedschaft: Individualereignis
    - Institution
    - Rollen (inklusive Status wie "Kandidat", "Mitglied", "Vorstand" etc. mit Zeitangaben)
    - Art der Beendigung

    Du extrahierst Informationen zu '{selected_category}'.
    """

    system_prompt = base_prompt + f"\n- {category_prompts(selected_category)}"

    if "Groq" in model:
        response = groq_response(model, system_prompt, user_question)
    elif "Google" in model:
        response = google_response(model, system_prompt, user_question)

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
    Wähle einen Ereignistyp aus, zu dem du Attribute aus einem Lexikonartikel 
    extrahieren möchtest:
    """
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Select model
    model_options = [
        'Groq: Mixtral 8x7b', 
        'Groq: Llama3-8B',
        'Groq: Llama3-70B',
        'Google: Gemini Flash 1.5']
    
    model = st.sidebar.selectbox("Select Model:", model_options, index=0)

    # Radio button options for categories
    categories = [
        "Lebensdaten, Familie und persönliches Netzwerk", 
        "Beruf und Ausbildung", 
        "Mitgliedschaften", 
        "Lebensorte",
        "Würdigung",
        "Sonstiges"
        ]
    
    selected_category = st.radio("Ereignistyp auswählen:", categories)

    st.markdown(multiline_text, unsafe_allow_html=True)

    # Get the user's question
    user_question = st.text_area(
        "Text Box to paste / input user text",
        label_visibility = 'hidden',
        placeholder="Füge hier deinen Text ein...")

    if user_question:
        response = information_extraction(client, model, user_question, selected_category)
        st.write(response)


if __name__ == "__main__":
    main()
