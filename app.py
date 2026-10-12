import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. Page Configuration (Wide Layout for a dashboard feel)
st.set_page_config(page_title="AI Utility Bill Decoder", page_icon="⚡", layout="wide")

# 2. Custom CSS for a premium, professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Style the main action button */
    div.stButton > button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Style the info boxes */
    div.stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown('<div class="main-header">⚡ AI Utility Bill Decoder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract structured data and answer complex queries from utility bills using Vision AI</div>', unsafe_allow_html=True)
st.divider()

# API Key Configuration
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.sidebar.error("⚠️ API Key is missing! Please configure Streamlit secrets.")

# 4. Professional Dashboard Layout (Two Columns)
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📁 1. Upload Document")
    uploaded_file = st.file_uploader("Upload a high-resolution image of the bill (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Document Preview", use_column_width=True)
    else:
        st.info("💡 Please upload a utility bill to get started.")

with col2:
    st.subheader("🤖 2. AI Analysis & Q&A")
    
    if uploaded_file is not None:
        st.write("Ask any specific question about the bill, or leave it blank for a full JSON extraction.")
        
        # User Input
        question = st.text_input("Enter your query:", placeholder="e.g., What is the total payable amount? or Extract data to JSON")
        
        # Action Button
        if st.button("🚀 Analyze & Decode"):
            with st.spinner("AI is analyzing the document... Please wait."):
                try:
                    model = genai.GenerativeModel('gemini-3.5-flash-lite')
                    # Default prompt if user asks nothing
                    prompt = question if question else "Analyze this utility bill and extract all key details (like names, dates, amounts, taxes, and units) into a strictly formatted JSON structure."
                    
                    response = model.generate_content([prompt, image])
                    
                    st.success("✅ Analysis Complete!")
                    st.markdown("### 📊 Results:")
                    # Displaying results in a clean box
                    st.info(response.text) 
                except Exception as e:
                    st.error(f"⚠️ An error occurred during analysis: {e}")
    else:
        st.warning("👈 Upload a document on the left panel to activate the AI assistant.")

# 5. Footer
st.markdown("<br><hr><center><small>Developed for WomenTechQuest (WTQ) Build Track Hackathon | Powered by Google Gemini</small></center>", unsafe_allow_html=True)