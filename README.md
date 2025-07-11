# 🔍 RAG Search Engine: Intelligent Document Querying Using AWS and Langchain

A powerful Retrieval-Augmented Generation (RAG) search engine that enables intelligent querying of unstructured documents using **LangChain**, **AWS Bedrock**, and **FAISS**. Built with a clean and interactive **Streamlit** frontend.

![RAG Workflow](https://your-image-link.com) <!-- optional banner or workflow image -->

---

## 🚀 Features

- 📄 Ingest unstructured data (PDFs, reports)
- 🧠 Integrate with AWS Bedrock LLMs (Claude 2.1, Llama 3)
- 🔗 LangChain for chaining, orchestration, and retrieval
- 📊 FAISS vector store for similarity search
- 🌐 Streamlit UI for interactive querying
- 🔒 Grounded responses with minimal hallucination

---

## 🧱 Tech Stack

| Layer         | Technology                       |
|--------------|-----------------------------------|
| Orchestration| [LangChain](https://www.langchain.com/) |
| Embeddings   | Amazon Titan (via Bedrock)        |
| Vector DB    | FAISS                             |
| LLMs         | Claude 2.1, Llama 3 via Bedrock   |
| Frontend     | Streamlit                         |
| Hosting      | AWS (Optional)                    |

---
## Project Result

![Screenshot (263)](https://github.com/user-attachments/assets/7b150ebd-8def-4d77-a132-708111df7330)
![Screenshot (265)](https://github.com/user-attachments/assets/98ace435-6649-454a-a2c2-8368b5ccd676)
![Screenshot (264)](https://github.com/user-attachments/assets/6f585856-e17d-43dc-94ce-b789ae12dbd3)

## 📂 Project Structure

```bash
├── app.py                  # Streamlit UI
├── ingest.py               # PDF ingestion & vector store creation
├── .env                    # AWS credentials & config
├── requirements.txt        # Python dependencies
├── data/                   # Folder for PDF documents
└── vectorstore/            # FAISS index

