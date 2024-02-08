import streamlit as st

def main():
    st.title("Meine erste Streamlit-Anwendung")

    # Benutzereingabe
    user_input = st.text_input("Gib deinen Namen ein:")

    # Verarbeitung der Benutzereingabe
    if user_input:
        greeting = f"Hallo, {user_input}! Willkommen zu Streamlit."
        st.write(greeting)

if __name__ == "__main__":
    main()
