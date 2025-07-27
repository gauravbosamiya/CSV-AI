import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.summarize import summarize
from src.home_page import home_page
from src.analyze import analyze

def main():
    st.set_page_config(page_title="CSV AI", page_icon="🧾", layout="wide")
    st.sidebar.title("🔐 API Configuration")

    st.markdown("""
        <div style='text-align: center;'>
            <h1>🧠 CSV AI</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center;'>
            <h4>⚡️ Interacting, Analyzing and Summarizing CSV Files!</h4>
        </div>
    """, unsafe_allow_html=True)

    if "api_loaded" not in st.session_state:
        st.session_state.api_loaded = False

    user_api_key = st.sidebar.text_input(
        label="#### Enter Google API key 👇",
        placeholder="Paste your Google API key",
        type="password"
    )

    if st.sidebar.button("Load"):
        if user_api_key:
            st.session_state.api_loaded = True
            st.session_state.api_key = user_api_key
            st.sidebar.success("API key loaded", icon="🚀")
        else:
            st.warning("Please enter your Google API key to continue.")

    MODEL_OPTION = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTION)
    api_key = st.session_state.get("api_key", "")

    functions = [
        "home",
        "Chat with CSV",
        "Summarize CSV",
        "Analyze CSV",
    ]
    selected_function = st.selectbox("Select a functionality", functions)

    if selected_function == "home":
        st.write("👋 Welcome! Please upload a CSV or Excel file to get started.")
        home_page()

    elif not st.session_state.api_loaded:
        st.warning("⚠️ Please enter your Google API key to access this feature.")
        st.stop()

    else:
        try:
            if selected_function == "Chat with CSV":
                model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                result = model.invoke("Hello!")
                st.write("Response:", result.content)

            elif selected_function == "Summarize CSV":
                summarize(model_name, api_key)  

            elif selected_function == "Analyze CSV":
                analyze(model_name, api_key)

        except Exception as e:
            st.error("❌ Invalid API key. Please enter a valid **Google Generative AI** key.")
            st.exception(e)
            st.stop()


if __name__ == "__main__":
    main()


