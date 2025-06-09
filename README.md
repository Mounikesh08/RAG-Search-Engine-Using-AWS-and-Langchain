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

## 📂 Project Structure

```bash
├── app.py                  # Streamlit UI
├── ingest.py               # PDF ingestion & vector store creation
├── .env                    # AWS credentials & config
├── requirements.txt        # Python dependencies
├── data/                   # Folder for PDF documents
└── vectorstore/            # FAISS index
