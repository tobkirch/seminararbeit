import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from github import Github
from io import StringIO
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

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
    model_path = "mnv2_model"  # Modell im selben Ordner wie main.py
    st.session_state['model'] = tf.keras.models.load_model(model_path)
    
st.title('Bildklassifizierung Werkzeugverschleiß')
tab1, tab2 = st.tabs(["Vorhersage tätigen", "Gespeicherte Daten"])

# Streamlit-Anwendung
def main():
    with tab1:   
        st.header("Schritt 1: Bild auswählen")
        st.write("Wähle das Bild aus für das eine Vorhersage getätigt werden soll. Hierfür bestehen zwei Möglichkeiten:")
        # Bild hochladen
        tab3, tab4 = st.tabs(["Bild hochladen", "Bild aufnehmen"])
        with tab3:
            st.write("Lade das Bild einer Wendeschneidplatte hoch")
            uploaded_image = st.file_uploader("Wenn du bereits ein Bild mit der Kamera aufgenommen hast, wird es hierduch ersetzt", type=['jpg', 'jpeg', 'png'])
        with tab4:
            if uploaded_image is None:
                camera_image = st.camera_input(" ")
            else:
                st.info("Entferne erst das hochgeladene Bild, bevor du hier eines mit deiner Kamera aufnehmen kannst")
        
        if uploaded_image is not None or camera_image is not None:
            # Bild zuschneiden
            st.divider()
            st.header("Schritt 2: Bild zuschneiden")
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
            else:
                image = Image.open(camera_image)
            
            st.write("Schneide das Bild auf die Obere Kante und Schneidecke zu")
            c1, c2 = st.columns([25,75])
            with c1:
                image = crop_image(image)
            with c2:
                st.image(image, caption="Zugeschnittenes Bild", use_column_width=True)
            
            # Button zum Vorhersagen
            st.divider()
            st.header('Schritt 3: Vorhersage tätigen')
            st.write("Klicke hier um eine Vorhersage für das ausgewählte Bild zu tätigen:")
            c3, c4= st.columns([30, 70])
            with c3:
                if st.button('Vorhersage tätigen'):
                        # Vorhersage mit dem Modell
                        st.session_state['prediction'] = predict(image)
            with c4:
                if st.session_state.prediction is None:
                    st.info('Vorhersage des Modells: ...')
                elif st.session_state.prediction == "Defekt":
                    st.error('Defekt')
                elif st.session_state.prediction == "Mittel":
                    st.warning('Mittel')
                elif st.session_state.prediction == "Neuwertig":
                    st.success('Neuwertig')
    
            # Zusätzliche Bauteildaten
            if st.session_state.prediction is not None:
                st.divider()
                st.header('Schritt 4: Vorhersage speichern')
                if st.session_state.show is True:
                    # Textfeldeingaben
                    st.write("Gib zusätzliche Daten über die Wendeschneidplatte an um sie mit der Vorhersage zu speichern")
                    # Variablen für die CSV-Eingabe
                    bauteil_name = st.text_input("Name des Bauteils")
                    werkzeugtyp = st.text_input("Werkzeugtyp")
                    vorschub = st.text_input("Vorschub")
                    drehzahl = st.text_input("Drehzahl")
                    zustellung = st.text_input("Zustellung")
                    bearbeitungsdauer = st.text_input("Bearbeitungsdauer")
                    
                    # Speichern Button
                    if st.button("Daten Speichern"):
    
                        # Laden der bisherigen Daten von GitHub
                        g = Github(github_token)
                        repo = g.get_repo(f"{github_repo_owner}/{github_repo_name}")
                        contents = repo.get_contents(github_file_path)
                        csv_content = contents.decoded_content.decode('utf-8')
                        existing_df = pd.read_csv(StringIO(csv_content))
                        
                        new_data = {"Name des Bauteils": [bauteil_name], "Werkzeugtyp": [werkzeugtyp], "Vorschub": [vorschub], "Drehzahl": [drehzahl], "Zustellung": [zustellung], "Bearbeitungsdauer": [bearbeitungsdauer], "Vorhersage": [st.session_state.prediction]}
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
            st.session_state.prediction = None
            st.session_state.show = True
            st.session_state.saved = False

    with tab2:
        st.header("Deine gespeicherten Daten")
        g = Github(github_token)
        repo = g.get_repo(f"{github_repo_owner}/{github_repo_name}")
        contents = repo.get_contents(github_file_path)
        csv_content = contents.decoded_content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        df_show = df
        tab5, tab6 = st.tabs(["Tabelle", "Diagramm"])
        with tab5:
            st.write("Hier werden deine gespeicherten Daten als Tabelle angezeigt.")
            st.write("Über die Filterung lässt sich steuern welche Einträge angezeigt werden sollen.")
            # Filterfunktion hinzufügen
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                search_column = st.selectbox("Spalte", df.columns)
            with col2:
                search_query = st.text_input("Suchwort")
            with col3:
                st.write('<div style="height: 28px;"></div>', unsafe_allow_html=True)
                search_button = st.button("Tabelle filtern")
            with col4:
                showAll_button = st.button("Alles anzeigen")
            if search_button:
                df_show = df[df[search_column].str.contains(search_query, case=False)]
            else:
                df_show = df
            if showAll_button:
                df_show = df
            st.write(df_show)
        
        with tab6:
            # Verschleißverlauf über die Zeit anzeigen
            st.write("Hier werden deine gespeicherten Daten aus Diagramm angezeigt, dass die Vorhersage des Verschleißgrades über die Bearbeitungszeit aufführt.")
            if not df.empty:
                df["Bearbeitungsdauer"] = pd.to_numeric(df["Bearbeitungsdauer"], errors='coerce')
                df["Vorhersage"] = df["Vorhersage"].astype('category')
            
                # Dropdown-Liste für die Auswahl der Bauteile hinzufügen
                selected_parts = st.multiselect("Anzuzeigende Bauteile auswählen", df["Name des Bauteils"].unique(), default=df["Name des Bauteils"].unique())
            
                # Erstellen des Diagramms mit Matplotlib oder Seaborn
                fig, ax = plt.subplots(figsize=(10, 3))
            
                # Gruppierung nach Bauteilnamen und Erstellung von Plots für ausgewählte Gruppen
                grouped = df.groupby("Name des Bauteils")
                for name, group in grouped:
                    if name in selected_parts:
                        chart_data = group[["Bearbeitungsdauer", "Vorhersage"]]
                        chart_data = chart_data.sort_values(by="Bearbeitungsdauer")  # Sortierung nach Bearbeitungsdauer
            
                        # Plotten des Verschleißverlaufs für ausgewählte Bauteile
                        sns.lineplot(data=chart_data, x="Bearbeitungsdauer", y="Vorhersage", ax=ax, label=name)
            
                # Hilfslinien hinzufügen
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
                # Achsenbeschriftungen und Titel hinzufügen
                ax.set_title("Verschleißverlauf über die Zeit")
                ax.set_xlabel("Bearbeitungsdauer")
                ax.set_ylabel("Vorhersage")
                ax.tick_params(axis='x')
                ax.legend(title="Bauteil")
            
                # Diagramm anzeigen
                st.pyplot(fig)
    
def crop_image (image):
    # Zuschnittbereich auswählen
    st.write("Verschiebe die Regler um das Bild zuzuschneiden")
    left = st.slider("Linker Rand:", 0, image.width, 0)
    top = st.slider("Oberer Rand:", 0, image.height, 0)
    right = st.slider("Rechter Rand:", 0, image.width, image.width)
    bottom = st.slider("Unterer Rand:", 0, image.height, image.height)
    
    # Bild zuschneiden
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image
    
def predict(img):
    class_names = ["Defekt", "Mittel", "Neuwertig"]
    # Laden des Bildes und Umwandeln in das richtige Format
    img = img.resize((224, 224))  # Skalieren des Bildes auf die gewünschte Größe
    img = np.array(img) / 255.0  # Normalisieren des Bildes
    
    # Hinzufügen einer zusätzlichen Dimension, um eine Batch-Dimension zu simulieren
    img = np.expand_dims(img, axis=0)
    
    # Vorhersage durchführen
    predictions = st.session_state.model.predict(img)
    
    # Extrahieren der wahrscheinlichsten Klasse
    predicted_class = np.argmax(predictions[0])
    
    # Rückgabe der vorhergesagten Klasse
    return class_names[predicted_class]
    
if __name__ == '__main__':
    main()
