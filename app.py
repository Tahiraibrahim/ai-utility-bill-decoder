import streamlit as st
import google.generativeai as genai
from PIL import Image

# App ka Title
st.set_page_config(page_title="AI Utility Bill Decoder", page_icon="⚡")
st.title("⚡ AI Utility Bill Decoder")
st.write("Apna electricity bill upload karein aur AI se koi bhi sawal pochein!")

# API Key ko Streamlit secrets se configure karna
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("API Key secrets mein nahi mili. Kripya Streamlit settings check karein.")

# Image uploader
uploaded_file = st.file_uploader("Bill ki tasveer upload karein (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Image dikhana
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Bill", use_column_width=True)

    # User se sawal lena
    question = st.text_input("Bill ke baare mein sawal likhein (Misaal: Total amount kya hai? ya JSON extract karo):")

    if st.button("Decode Bill"):
        with st.spinner("AI bill parh raha hai..."):
            try:
                # Gemini model call
                model = genai.GenerativeModel('gemini-3.5-flash-lite')
                prompt = question if question else "Please extract the key details of this bill into JSON format."
                
                response = model.generate_content([prompt, image])
                
                st.success("Analysis Complete!")
                st.write(response.text)
            except Exception as e:
                st.error(f"Ek error aagaya: {e}")