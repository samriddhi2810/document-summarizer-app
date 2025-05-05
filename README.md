# 📄 Document Summarizer App

An intelligent PDF summarization app that leverages Transformer-based NLP models to generate concise and structured summaries from lengthy PDF documents.

> 🔍 Built with [Streamlit](https://streamlit.io), [HuggingFace Transformers](https://huggingface.co/transformers/), and a custom fine-tuned T5 model.

---

## 🚀 Features

✅ Upload any `.pdf` file  
✅ Extracts meaningful content using `pdfplumber`  
✅ Smart paragraph-based chunking (up to 4000 chars)  
✅ Generates summaries using `T5ForConditionalGeneration`  
✅ Live chunk-by-chunk summarization with progress bar  
✅ Beautiful bullet-point formatted summary  
✅ Download final summary as `.txt` file

---

## 📸 UI Preview

![Poster Overview](assets/poster.png)

---

## ⚙️ Tech Stack

| Layer          | Technology                     |
|----------------|--------------------------------|
| Frontend UI    | Streamlit                      |
| NLP Model      | HuggingFace T5 (local)         |
| Text Extraction| pdfplumber                     |
| Summarization  | Transformers Pipeline API      |
| Chunk Handling | LangChain Recursive Splitter   |
| Backend Engine | PyTorch                        |

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/document-summarizer-app.git
cd document-summarizer-app
