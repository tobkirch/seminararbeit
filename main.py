import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pickle

# Laden des vorher trainierten Modells
#model = pickle.load(open('model.sav', 'rb'))

# Funktion zum Hochladen einer CSV-Datei
def upload_csv():
    uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=";")
    return None

# Funktion zum Speichern von Daten in der CSV-Datei
def save_to_csv(data, filename):
    data.to_csv(filename, index=False, sep=";")

# Streamlit-Anwendung
def main():
    st.title('Bildklassifizierung mit Machine Learning')
    
    st.write('Lade ein Bild hoch.')

    # Bild hochladen
    uploaded_image = st.file_uploader("Bild auswählen", type=['jpg', 'jpeg', 'png'])

    # CSV-Datei hochladen
    st.header("Lade eine CSV-Datei hoch um zusätzliche Infos über das Bauteil zu speichern")
    df = upload_csv()
    if df is not None:
        st.dataframe(df)
        # Variablen für die CSV-Eingabe
        werkzeugtyp = st.text_input('Werkzeugtyp')
        vorschub = st.number_input('Vorschub')
        drehzahl = st.number_input('Drehzahl')
        zustellung = st.number_input('Zustellung')
        bauteil_name = st.text_input('Name des Bauteils')
        bearbeitungsdauer = st.number_input('Bearbeitungsdauer (Minuten)')

    if uploaded_image is not None:
        # Bild anzeigen
        image = Image.open(uploaded_image)
        st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

        # Button zum Vorhersagen
        if st.button('Vorhersage machen'):
            # Vorhersage mit dem Modell
            #prediction = predict_image(np.array(image))
            prediction = 'Test'

            # Ergebnis anzeigen
            st.write('Das Bild zeigt:', prediction)
            
    if df is not None:
        # Button zum Speichern der Daten
        if st.button("Daten speichern"):
            new_entry = {'Spalte 1': text_input1, 'Spalte 2': text_input2}
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            save_to_csv(df, "updated_data.csv")
            st.success("Daten erfolgreich gespeichert!")
            
        # Button zum Herunterladen der CSV-Datei
        st.header("CSV-Datei herunterladen")
        with open("updated_data.csv", "rb") as file:
            st.download_button(label="Klicke hier, um die aktualisierte CSV-Datei herunterzuladen",
                                data=file,
                                file_name="updated_data.csv",
                                mime="text/csv")


def predict_image(image):
    # Hier sollte der Code stehen, um das Bild für das Modell vorzubereiten
    # ...

    # Platzhalter für die Vorhersage
    prediction = "Platzhalter-Vorhersage"
    return prediction

def save_to_csv(werkzeugtyp, vorschub, drehzahl, zustellung, bauteil_name, bearbeitungsdauer, prediction):
    try:
        # DataFrame erstellen
        data = {
            'Werkzeugtyp': [werkzeugtyp],
            'Vorschub': [vorschub],
            'Drehzahl': [drehzahl],
            'Zustellung': [zustellung],
            'Bauteil Name': [bauteil_name],
            'Bearbeitungsdauer (Minuten)': [bearbeitungsdauer],
            'Vorhersage': [prediction]
        }
        df = pd.DataFrame(data)
    
        # CSV-Datei speichern
        df.to_csv('ergebnisse.csv', mode='a', header=False, index=False)
        
        return True
    except Exception as e:
        print("Fehler beim Speichern der CSV-Datei:", e)
        return False

if __name__ == '__main__':
    main()
