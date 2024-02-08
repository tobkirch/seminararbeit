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

# Streamlit-Anwendung
def main():
    st.title('Bildklassifizierung mit Machine Learning')
    
    st.header('Lade ein Bild hoch.')

    # Bild hochladen
    uploaded_image = st.file_uploader("Bild auswählen", type=['jpg', 'jpeg', 'png'])
    
    prediction = None  # Initialisiere prediction mit None
    
    if uploaded_image is not None:
        # Bild anzeigen
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

        # Zusätzliche Bauteildaten
        st.header("Zusätzliche Daten")
        
        
        # Button zum Vorhersagen
        if st.button('Vorhersage machen'):
            # Vorhersage mit dem Modell
            prediction = predict_image(np.array(image))
            # Ergebnis anzeigen
            st.success('Das Bauteil ist: '+ prediction)
            
    if prediction is not None:
        # Button zum Speichern der Daten
        # Textfeldeingaben
        input1 = st.text_input("Eingabe 1")
        input2 = st.text_input("Eingabe 2")
            
        # Laden der bisherigen Daten von GitHub
        g = Github(github_token)
        repo = g.get_repo(f"{github_repo_owner}/{github_repo_name2}")
        contents = repo.get_contents(github_file_path)
        csv_content = contents.decoded_content.decode('utf-8')
        existing_df = pd.read_csv(StringIO(csv_content))
            
        # Speichern Button
        if st.button("Speichern"):
            # Neue Daten hinzufügen
            new_data = {"Eingabe 1": [input1], "Eingabe 2": [input2]}
            new_df = pd.DataFrame(new_data)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                
            # CSV Datei auf GitHub aktualisieren
            repo.update_file(contents.path, "Daten aktualisiert", updated_df.to_csv(index=False), contents.sha)
                
            st.success("Daten erfolgreich gespeichert!")

def predict_image(image):
    # Hier sollte der Code stehen, um das Bild für das Modell vorzubereiten
    # ...

    # Platzhalter für die Vorhersage
    prediction = "Platzhalter-Vorhersage"
    return prediction

def save_to_csv(new_df):
    print("Speichern der Daten...")
    # Neue Daten hinzufügen
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # CSV-Datei auf GitHub aktualisieren
    csv_data = updated_df.to_csv(index=False)
    updated_file_content = StringIO(csv_data).read()
    print("Aktualisierter Dateiinhalt:", updated_file_content)
    print("Aktualisieren der Datei auf GitHub...")
    try:
        repo.update_file(contents.path, "Daten aktualisiert", updated_file_content, contents.sha)
        st.success("Daten erfolgreich gespeichert!")
        print("Daten erfolgreich gespeichert!")
    except Exception as e:
        st.error("Fehler beim Speichern der Daten: " + str(e))
        print("Fehler beim Speichern der Daten:", e)


    
if __name__ == '__main__':
    main()
