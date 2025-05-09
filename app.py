import warnings
warnings.filterwarnings("ignore")

import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import pdfplumber

# Load Model and Tokenizer
checkpoint = "C:/Users/samri/OneDrive/Desktop/AIPROJECT/lamini-lm"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function to display PDF in Streamlit
@st.cache_data()
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to prettify bullet summary
def prettify_summary(summary_text):
    sentences = summary_text.split('. ')
    pretty_summary = ""
    for sentence in sentences:
        if sentence.strip() != "":
            pretty_summary += f"• {sentence.strip()}.\n\n"
    return pretty_summary

# LangChain-based summarization pipeline
def llm_pipeline(filepath):
    # Load PDF file using LangChain's PyPDFLoader
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    # Split documents into readable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Use HF pipeline wrapped in LangChain
    summarizer = pipeline("summarization", model=base_model, tokenizer=tokenizer)
    hf = HuggingFacePipeline(pipeline=summarizer)

    # Load LangChain summarization chain (map_reduce = summary-of-summaries)
    chain = load_summarize_chain(hf, chain_type="map_reduce")

    # Visual feedback
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run summarization
    summary = chain.run(docs)
    progress_bar.progress(1.0)
    status_text.text("✅ Summarization Completed!")

    # Stats
    full_text = " ".join([doc.page_content for doc in docs])
    original_characters = len(full_text)
    summarized_characters = len(summary)
    compression = (summarized_characters / original_characters) * 100 if original_characters > 0 else 0

    return summary, original_characters, summarized_characters, compression

# Streamlit Page Layout
st.set_page_config(layout='wide')

# UI Custom Styling
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
}
.stApp {
    background-color: #0F172A;
    background-image: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #00FFFF;
}
h1, h2, h3, h4, h5, h6 {
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF;
}
.stMarkdown p {
    color: #E0FFFF;
}
.css-1d391kg {
    background: linear-gradient(135deg, #0F172A, #1E293B);
}
.stButton > button {
    background-color: transparent;
    border: 2px solid #00FFFF;
    color: #00FFFF;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    box-shadow: 0 0 15px #00FFFF;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: #00FFFF20;
    color: white;
    transform: scale(1.05);
}
.stExpander {
    background-color: #1E293B;
    border: 1px solid #00FFFF;
}
</style>
""", unsafe_allow_html=True)

# Main App Logic
def main():
    st.title('Document Summarization App using Language Model')
    uploaded_file = st.file_uploader("Upload your pdf file", type=['pdf'])

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with pdfplumber.open("temp.pdf") as pdf:
            page_count = len(pdf.pages)

        if file_size_mb > 5 or page_count > 200:
            st.warning(f"⚡ Warning: Large file detected! (~{file_size_mb:.2f} MB, {page_count} pages)\n\nSummarization might take a bit longer ⏳.")

        st.success(f"✅ Successfully uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB, {page_count} pages)")
        displayPDF("temp.pdf")

        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded PDF file")
                displayPDF("temp.pdf")

            with col2:
                st.info("Summarization Output")
                with st.spinner("Generating Summary..."):
                    summary, original_characters, summarized_characters, compression = llm_pipeline("temp.pdf")

                st.success("✅ Summarization Completed!")
                st.info(f"""
                **🧾 Character Statistics:**
                - 📑 Original Document Characters: {original_characters}
                - 📝 Summarized Document Characters: {summarized_characters}
                - 📉 Compression Achieved: {compression:.2f}%
                """)
                pretty_summary = prettify_summary(summary)
                st.write(pretty_summary)
                st.download_button(
                    label="Download Summary",
                    data=pretty_summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

if __name__ == '__main__':
    main()
