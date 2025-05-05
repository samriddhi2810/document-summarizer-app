import warnings
warnings.filterwarnings("ignore")

import streamlit as st 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline 
import torch
import base64
import pdfplumber

# Model and Tokenizer 
checkpoint = "C:/Users/samri/OneDrive/Desktop/AIPROJECT/lamini-lm"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# file loader and preprocessing 
def file_preprocessing(file):
    """
    Load PDF with pdfplumber and return full clean text.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def count_characters(text):
    return len(text)


# LM pipeline

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer
    )
    input_text = file_preprocessing(filepath)
    input_text = input_text.replace('\n', ' ')

    # 🧠 Count Original Characters
    original_character_count = count_characters(input_text)

    input_chunks = [input_text[i:i+500] for i in range(0, len(input_text), 500)]
    total_chunks = len(input_chunks)

    final_summary = ""

    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, chunk in enumerate(input_chunks):
        chunk_tokens = len(chunk.split())
        calculated_max_length = min(300, int(chunk_tokens * 2))
        calculated_min_length = max(30, int(chunk_tokens / 2))

        result = pipe_sum(
            chunk,
            max_length=calculated_max_length,
            min_length=calculated_min_length
        )
        final_summary += result[0]['summary_text'] + " "

        # Update progress
        progress_percentage = (idx + 1) / total_chunks
        progress_bar.progress(progress_percentage)
        status_text.text(f"Summarizing chunk {idx+1}/{total_chunks}... ({int(progress_percentage*100)}%)")

    # 🧠 Count Summary Characters
    summary_character_count = count_characters(final_summary)

    # 🧠 Calculate Compression %
    compression = (summary_character_count / original_character_count) * 100

    status_text.text("✅ Summarization Completed!")

    # 🧠 Return summary + counts
    return final_summary, original_character_count, summary_character_count, compression


def prettify_summary(summary_text):
    # Split summary into sentences
    sentences = summary_text.split('. ')
    # Add a bullet point before each sentence
    pretty_summary = ""
    for sentence in sentences:
        if sentence.strip() != "":
            pretty_summary += f"• {sentence.strip()}.\n\n"
    return pretty_summary


@st.cache_data()
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# streamlit code
st.set_page_config(layout='wide')

# Custom CSS for NEON Light Blue Text Button + Font Change
st.markdown(
    """
    <style>
    /* Set overall font */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
    }

    /* Change background */
    .stApp {
        background-color: #0F172A;
        background-image: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #00FFFF;
    }

    /* Make headings Neon Blue */
    h1, h2, h3, h4, h5, h6 {
        color: #00FFFF;
        text-shadow: 0 0 10px #00FFFF;
    }

    /* Normal text */
    .stMarkdown p {
        color: #E0FFFF;
    }

    /* Sidebar background (if you use) */
    .css-1d391kg {
        background: linear-gradient(135deg, #0F172A, #1E293B);
    }

    /* Summarize Button Custom */
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

    /* Expander style (optional) */
    .stExpander {
        background-color: #1E293B;
        border: 1px solid #00FFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    import os

def main():
    st.title('Document Summarization App using Language Model')

    uploaded_file = st.file_uploader("Upload your pdf file", type=['pdf'])

    if uploaded_file is not None:
        # 🛠 Check file size (in MB)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        # 🛠 Save the uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # 🛠 Check page count
        import pdfplumber
        with pdfplumber.open("temp.pdf") as pdf:
            page_count = len(pdf.pages)

        # 🛠 Show warnings if needed
        if file_size_mb > 5 or page_count > 200:
            st.warning(f"⚡ Warning: Large file detected! (~{file_size_mb:.2f} MB, {page_count} pages)\n\nSummarization might take a bit longer ⏳.")

        st.success(f"✅ Successfully uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB, {page_count} pages)")

        # Display uploaded PDF
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
