import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pickle
from github import Github
from io import StringIO
import tensorflow as tf

# GitHub Zugangsdaten
github_token = st.secrets["GH_Token"]
github_repo_owner = "tobkirch"
github_repo_name = "seminararbeit"
github_repo_name2 = "test"
github_file_path = "ergebnisse.csv"

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'show' not in st.session_state:
    st.session_state['show'] = True

if 'saved' not in st.session_state:
    st.session_state['saved'] = False

# Laden des vorher trainierten Modells
if 'model' not in st.session_state:
    st.session_state['model'] = tf.keras.models.load_model('mnv2_model')
    
# Streamlit-Anwendung
def main():
    st.title('Bildklassifizierung Werkzeugverschleiß')
    
    st.header('Bild hochladen')

    # Bild hochladen
    uploaded_image = st.file_uploader("Lade das Bild einer Wendeschneidplatte hoch", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # Bild anzeigen
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        st.write()
        
        # Button zum Vorhersagen
        st.write("Klicke hier um eine Vorhersage für das ausgewählte Bild zu tätigen:")
        if st.button('Vorhersage tätigen'):
                # Vorhersage mit dem Modell
                st.session_state['prediction'] = predict(image)

        if st.session_state.prediction is None:
            st.info("Vorhersage des Modells: ...")
        elif st.session_state.prediction == 0:
            st.error("Defekt")
        elif st.session_state.prediction == 1:
            st.warning("Mittel")
        elif st.session_state.prediction == 2:
            st.success("Neuwertig")

        # Zusätzliche Bauteildaten
        st.header("Vorhersage speichern")
        
        if st.session_state.prediction is not None:
            if st.session_state.show is True:
                # Textfeldeingaben
                st.write("Gib zusätzliche Daten über die Wendeschneidplatte an um sie mit der Vorhersage zu speichern")
                # Variablen für die CSV-Eingabe
                werkzeugtyp = st.text_input("Werkzeugtyp")
                vorschub = st.text_input("Vorschub")
                drehzahl = st.text_input("Drehzahl")
                zustellung = st.text_input("Zustellung")
                bauteil_name = st.text_input("Name des Bauteils")
                bearbeitungsdauer = st.text_input("Bearbeitungsdauer")
                
                # Speichern Button
                if st.button("Daten Speichern"):

                    # Laden der bisherigen Daten von GitHub
                    g = Github(github_token)
                    repo = g.get_repo(f"{github_repo_owner}/{github_repo_name}")
                    contents = repo.get_contents(github_file_path)
                    csv_content = contents.decoded_content.decode('utf-8')
                    existing_df = pd.read_csv(StringIO(csv_content))
                    
                    new_data = {"Werkzeugtyp": [werkzeugtyp], "Vorschub": [vorschub], "Drehzahl": [drehzahl], "Zustellung": [zustellung], "Name des Bauteils": [bauteil_name], "Bearbeitungsdauer": [bearbeitungsdauer], "Vorhersage": [st.session_state.prediction]}
                    new_df = pd.DataFrame(new_data)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                    # CSV Datei auf GitHub aktualisieren
                    repo.update_file(contents.path, "Daten aktualisiert", updated_df.to_csv(index=False), contents.sha)
                    st.session_state.saved = True
                    st.session_state.show = False
                    st.rerun()
            if st.session_state.saved is True:
                st.success("Daten gespeichert!")
        else:
            st.write("Sobald eine Vorhersage getätigt wurde kann diese hier mit zusätzlichen Werkzeugdaten gespeichert werden")
    else:
        st.session_state.prediction = None
        st.session_state.show = True
        st.session_state.saved = False


def predict_image(image):
    # Hier sollte der Code stehen, um das Bild für das Modell vorzubereiten
    # ...

    # Platzhalter für die Vorhersage
    prediction = "Platzhalter-Vorhersage"
    return prediction

def predict(img):
    # Laden des Bildes und Umwandeln in das richtige Format
    img = img.resize((224, 224))  # Skalieren des Bildes auf die gewünschte Größe
    img = np.array(img) / 255.0  # Normalisieren des Bildes

    # Hinzufügen einer zusätzlichen Dimension, um eine Batch-Dimension zu simulieren
    img = np.expand_dims(img, axis=0)

     # Vorhersage durchführen
    predictions = mnv2_model.predict(img)

    # Extrahieren der wahrscheinlichsten Klasse
    predicted_class = np.argmax(predictions[0])

    # Rückgabe der vorhergesagten Klasse
    return predicted_class

if __name__ == '__main__':
    main()

