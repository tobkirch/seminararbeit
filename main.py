import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pickle

# Laden des vorher trainierten Modells
#model = pickle.load(open('model.sav', 'rb'))

# Streamlit-Anwendung
def main():
    st.title('Bildklassifizierung mit Machine Learning')
    st.write('Lade ein Bild hoch und gib die entsprechenden Variablen ein.')

    # Bild hochladen
    uploaded_image = st.file_uploader("Bild auswählen", type=['jpg', 'jpeg', 'png'])

    # Variablen für die CSV-Eingabe
    werkzeugtyp = st.text_input('Werkzeugtyp')
    vorschub = st.number_input('Vorschub', value=0.0)
    drehzahl = st.number_input('Drehzahl', value=0.0)
    zustellung = st.number_input('Zustellung', value=0.0)
    bauteil_name = st.text_input('Name des Bauteils')
    bearbeitungsdauer = st.number_input('Bearbeitungsdauer (Minuten)', value=0)

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

             # CSV-Datei speichern
            saved = save_to_csv(werkzeugtyp, vorschub, drehzahl, zustellung, bauteil_name, bearbeitungsdauer, prediction)
            if saved:
                st.write("CSV-Datei erfolgreich aktualisiert.")
            else:
                st.write("Fehler beim Aktualisieren der CSV-Datei.")

def predict_image(image):
    # Hier sollte der Code stehen, um das Bild für das Modell vorzubereiten
    # ...

    # Platzhalter für die Vorhersage
    prediction = "Platzhalter-Vorhersage"
    return prediction

def save_to_csv(werkzeugtyp, vorschub, drehzahl, zustellung, bauteil_name, bearbeitungsdauer, prediction):
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
    with open('ergebnisse.csv', mode='a') as file:
            df.to_csv(file, header=False, index=False)
            return True
    except Exception as e:
        print("Fehler beim Speichern der CSV-Datei:", e)
        return False

if __name__ == '__main__':
    main()
