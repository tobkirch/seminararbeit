import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pickle
from github import Github
from io import StringIO

# Laden des vorher trainierten Modells
#model = pickle.load(open('model.sav', 'rb'))

# GitHub Zugangsdaten
github_token = st.secrets["GH_Token"]
github_repo_owner = "tobkirch"
github_repo_name = "seminararbeit"
github_repo_name2 = "test"
github_file_path = "ergebnisse.csv"

# Laden der bisherigen Daten von GitHub
g = Github(github_token)
repo = g.get_repo(f"{github_repo_owner}/{github_repo_name}")
contents = repo.get_contents(github_file_path)
csv_content = contents.decoded_content.decode('utf-8')
existing_df = pd.read_csv(StringIO(csv_content))

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
    
# Streamlit-Anwendung
def main():
    st.title('Bildklassifizierung mit Machine Learning')
    
    st.header('Lade ein Bild hoch.')

    # Bild hochladen
    uploaded_image = st.file_uploader("Bild auswählen", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # Bild anzeigen
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        st.write()
        
        # Button zum Vorhersagen
        st.write("Klicke hier um eine Vorhersage für das ausgewählte Bild zu tätigen:")
        if st.button('Vorhersage tätigen'):
            # Vorhersage mit dem Modell
            st.session_state['prediction'] = predict_image(np.array(image))
            # Ergebnis anzeigen
            
        if prediction is "Platzhalter-Vorhersage":
            st.success(st.session_state.prediction)
        else:
            st.info("Vorhersage des Modells: ...")
                
        if st.session_state.prediction is not None:
            # Zusätzliche Bauteildaten
            st.header("Vorhersage speichern")
            # Textfeldeingaben
            st.write("Gib zusätzliche Daten über das Bauteil an um sie mit der Vorhersage zu speichern")
            # Variablen für die CSV-Eingabe
            werkzeugtyp = st.text_input("Werkzeugtyp")
            vorschub = st.text_input("Vorschub")
            drehzahl = st.text_input("Drehzahl")
            zustellung = st.text_input("Zustellung")
            bauteil_name = st.text_input("Name des Bauteils")
            bearbeitungsdauer = st.text_input("Bearbeitungsdauer")
            
            # Speichern Button
            if st.button("Daten Speichern"):
                new_data = {"Werkzeugtyp": [werkzeugtyp], "Vorschub": [vorschub], "Drehzahl": [drehzahl], "Zustellung": [zustellung], "Name des Bauteils": [bauteil_name], "Bearbeitungsdauer": [bearbeitungsdauer], "Vorhersage": [st.session_state.prediction]}
                new_df = pd.DataFrame(new_data)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                # CSV Datei auf GitHub aktualisieren
                repo.update_file(contents.path, "Daten aktualisiert", updated_df.to_csv(index=False), contents.sha)
                st.success("Daten erfolgreich gespeichert!")
                st.session_state['prediction'] = None
                    


def predict_image(image):
    # Hier sollte der Code stehen, um das Bild für das Modell vorzubereiten
    # ...

    # Platzhalter für die Vorhersage
    prediction = "Platzhalter-Vorhersage"
    return prediction

if __name__ == '__main__':
    main()

