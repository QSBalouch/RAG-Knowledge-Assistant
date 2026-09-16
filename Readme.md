
# 🩺 MedicalBot — Document-Grounded Medical Q&A

> **MedicalBot** is a Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from a collection of medical documents.
>
> It combines **FAISS vector search**, **Hugging Face embeddings**, and a **DeepSeek language model through the Hugging Face Router** to generate document-grounded answers through an interactive **Streamlit** chat interface.

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
<img src="https://img.shields.io/badge/LangChain-RAG-green" alt="LangChain">
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-orange" alt="FAISS">
<img src="https://img.shields.io/badge/HuggingFace-LLM-yellow?logo=huggingface" alt="Hugging Face">
<img src="https://img.shields.io/badge/Streamlit-Chat%20UI-red?logo=streamlit" alt="Streamlit">

</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [How RAG Works](#-how-rag-works)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Hugging Face API Token](#-hugging-face-api-token)
- [Creating the Vector Database](#-creating-the-vector-database)
- [Document Retrieval](#-document-retrieval)
- [Hugging Face Router](#-hugging-face-router)
- [RAG Prompt](#-rag-prompt)
- [Running the Application](#-running-the-application)
- [Streamlit Interface](#-streamlit-interface)
- [Source References](#-source-references)
- [Example](#-example)
- [Out-of-Context Questions](#-out-of-context-questions)
- [Security](#-security)
- [Current Pipeline](#-current-pipeline)
- [Project Goal](#-project-goal)
- [Future Improvements](#-future-improvements)
- [Limitations](#-limitations)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🔎 Overview

MedicalBot demonstrates how to build a **document-grounded conversational AI system** using **Retrieval-Augmented Generation (RAG)**.

Instead of sending a user's question directly to a language model, MedicalBot first searches a collection of medical documents for relevant information.

The retrieved information is then provided to the language model as context. The model generates an answer based on the retrieved evidence.

### 🔄 Basic Workflow

```text
User Question
      │
      ▼
Streamlit Chat Interface
      │
      ▼
Question Embedding
      │
      ▼
FAISS Vector Search
      │
      ▼
Relevant Medical Documents
      │
      ▼
RAG Prompt
      │
      ▼
Hugging Face Router
      │
      ▼
DeepSeek LLM
      │
      ▼
Generated Answer
      │
      ▼
Answer + Sources
````

---

 ## ✨ Features

 - 📚 Ask questions about medical documents
- 🔎 Semantic document retrieval using FAISS
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤗 Hugging Face embeddings
- 🚀 Hugging Face Router for remote LLM inference
- ⚡ DeepSeek model for answer generation
- 💬 Interactive Streamlit chat interface
- 📄 Displays retrieved source documents
- 📑 Shows source page numbers and document paths
- 🚫 Attempts to prevent answers outside the provided documents
- 💾 Uses pre-generated embeddings instead of embedding documents at every query
- 🔐 Supports environment variables for API credentials

---

 ## 🏗️ Architecture

 The application follows this pipeline:

```
                    ┌─────────────────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Streamlit Chat UI  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  HuggingFace        │
                    │  Embeddings         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FAISS         │
                    │   Vector Search     │
                    └──────────┬──────────┘
                               │
                         Top-K Documents
                               │
                               ▼
                    ┌─────────────────────┐
                    │     RAG Prompt      │
                    │  Context + Query    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Hugging Face       │
                    │      Router         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     DeepSeek        │
                    │        LLM          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Answer + Sources   │
                    └─────────────────────┘
```

---

 ## 🧠 How RAG Works

 MedicalBot uses **Retrieval-Augmented Generation** instead of relying entirely on the language model's pre-trained knowledge.

 For example, suppose the user asks:

 > **What treatments are used for laryngeal cancer?**

 ### Step 1 — User Question

 The user submits the question through the Streamlit interface.

 ### Step 2 — Query Embedding

 The question is converted into a numerical vector using:

```
sentence-transformers/all-MiniLM-L6-v2
```

 ### Step 3 — FAISS Search

 FAISS searches the vector database for semantically similar document chunks.

 For example, it may retrieve information such as:

```
In the case of cancer of the larynx, radiotherapy is
the first choice to treat small lesions...
```

 ### Step 4 — Context Construction

 The retrieved documents are combined into a context section:

```
Context:
[Retrieved medical document content]

Question:
What treatments are used for laryngeal cancer?
```

 ### Step 5 — LLM Generation

 The context and question are sent to the DeepSeek model through the Hugging Face Router.

 ### Step 6 — Grounded Answer

 The model generates an answer based on the retrieved document context.

 This approach helps reduce reliance on unsupported information from the model's general knowledge.

---

 ## 🛠️ Technologies Used

 | Technology | Purpose |
| --- | --- |
| 🐍 Python | Core programming language |
| 🦜 LangChain | RAG and document-processing framework |
| 🔎 FAISS | Vector similarity search |
| 🤗 Hugging Face | Embeddings and remote model inference |
| 🧠 DeepSeek | Language model for answer generation |
| 🔤 Sentence Transformers | Document and query embeddings |
| 🎈 Streamlit | Web-based chat interface |
| 🌐 Requests | HTTP communication with Hugging Face Router |

---

 ## 📁 Project Structure

 A typical project structure looks like this:

```
MedicalBot/
│
├── data/
│   └── medical_documents.pdf
│
├── vector_store/
│   └── faiss_db/
│
├── create_memory.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

 ### `data/`

 Contains the source medical documents used to build the knowledge base.

 ### `vector_store/`

 Contains the FAISS vector database and associated embeddings.

 ### `create_memory.py`

 Responsible for:

 - Loading the FAISS database
- Retrieving relevant documents
- Constructing the RAG prompt
- Sending the prompt to the LLM
- Generating the response

 ### `app.py`

 Contains the Streamlit user interface and handles:

 - User questions
- Chat history
- Answer generation
- Retrieved source display

 ### `requirements.txt`

 Contains the Python dependencies required to run the project.

---

 ## 🚀 Installation

 ### 1\. Clone the Repository

```
git clone YOUR_GITHUB_REPOSITORY_URL
cd MedicalBot
```

 ### 2\. Create a Virtual Environment

 #### Windows

```
python -m venv venv
```

 Activate the environment:

```
venv\Scripts\activate
```

 #### Linux / macOS

```
python3 -m venv venv
source venv/bin/activate
```

 ### 3\. Install Dependencies

```
pip install -r requirements.txt
```

 If you do not have a `requirements.txt` file yet, the main packages include:

```
langchain
langchain-community
langchain-core
langchain-huggingface
faiss-cpu
sentence-transformers
transformers
requests
streamlit
```

---

 ## 🔑 Hugging Face API Token

 MedicalBot uses the **Hugging Face Router** to communicate with the remote language model.

 Create a Hugging Face API token through your Hugging Face account.

 Store the token as an environment variable instead of hard-coding it into your source code.

 ### Windows PowerShell

```
$env:HF_TOKEN="your_token_here"
```

 ### Python

```
import os

HF_TOKEN = os.getenv("HF_TOKEN")
```

 ### ❌ Do Not Do This

 Never hard-code your API token:

```
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
```

 ### ✅ Recommended

 Use an environment variable:

```
HF_TOKEN = os.getenv("HF_TOKEN")
```

 This prevents accidentally exposing credentials when the project is pushed to GitHub.

---

 ## 📚 Creating the Vector Database

 Before querying the system, the medical documents need to be converted into vector embeddings.

 The process is:

```
PDF / Documents
       │
       ▼
Extract Text
       │
       ▼
Split Into Chunks
       │
       ▼
Generate Embeddings
       │
       ▼
Store Embeddings
       │
       ▼
FAISS Vector Database
```

 MedicalBot uses:

```
sentence-transformers/all-MiniLM-L6-v2
```

 for generating embeddings.

 Once the vector database has been created, it can be loaded using:

```
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_store/faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
```

 > **Note:** Only use `allow_dangerous_deserialization=True` when loading a FAISS index from a source you trust.

---

 ## 🔎 Document Retrieval

 When a user submits a question, the question is converted into an embedding.

 FAISS then searches for document chunks with semantically similar embeddings.

 A retriever can be configured like this:

```
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)
```

 The parameter:

```
k=5
```

 means that up to five relevant document chunks are retrieved.

 These retrieved chunks are then provided to the language model as context.

 ### Retrieval Process

```
User Question
      │
      ▼
Question Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top-K Document Chunks
      │
      ▼
RAG Context
```

---

 ## 🤗 Hugging Face Router

 Instead of running a large language model locally, MedicalBot sends requests to the **Hugging Face Router**.

 The current implementation uses:

```
deepseek-ai/DeepSeek-V4.1-Flash
```

 The application communicates with the router using an HTTP request:

```
response = requests.post(
    HF_ROUTER_URL,
    headers=headers,
    json=payload,
    timeout=120
)
```

 The configured endpoint is:

```
https://router.huggingface.co/v1/chat/completions
```

 This allows the model to run remotely rather than requiring a large local GPU.

---

 ## 🧩 RAG Prompt

 MedicalBot uses a custom prompt to encourage the language model to answer only from the retrieved context.

 The basic concept is:

```
Answer using ONLY the provided context.

If the answer cannot be found in the context:

Answer not present in provided docs.

Do not use outside knowledge.
```

 The retrieved documents are inserted into a structure similar to:

```
Context:
{context}

Question:
{question}
```

 This creates the final prompt that is sent to the language model.

 ### Why This Matters

 The purpose of this restriction is to make the application **document-grounded**.

 Instead of:

```
Question → LLM → Answer
```

 the system follows:

```
Question
   ↓
Retrieve Evidence
   ↓
Evidence + Question
   ↓
LLM
   ↓
Grounded Answer
```

---

 ## ▶️ Running the Application

 After installing the dependencies and configuring the Hugging Face token, run:

```
streamlit run app.py
```

 Streamlit will start the application locally.

 The terminal will normally display a local address such as:

```
http://localhost:8501
```

 Open the displayed address in your web browser.

---

 ## 💬 Streamlit Interface

 The application provides an interactive conversational interface.

 Users can:

 - 💬 Submit medical questions
- 🤖 Receive generated answers
- 🗨️ Maintain chat history
- ⏳ See a loading indicator while the answer is generated
- 📄 Inspect retrieved source documents
- 📑 View document page numbers
- 📂 View document paths

 Example interface flow:

```
┌──────────────────────────────────────┐
│          🩺 MedicalBot               │
├──────────────────────────────────────┤
│                                      │
│ User: What treatments are used       │
│       for laryngeal cancer?          │
│                                      │
│ Bot: According to the provided       │
│      documents, treatments include   │
│      ...                              │
│                                      │
├──────────────────────────────────────┤
│ Sources                              │
│                                      │
│ 📄 Source 1 — Page 155               │
│ 📄 Source 2 — Page 587               │
│ 📄 Source 3 — Page 586               │
└──────────────────────────────────────┘
```

---

 ## 📄 Source References

 One of the important features of MedicalBot is the ability to display the documents retrieved during the RAG process.

 For example:

```
Sources

Source 1 — Page 155
Source 2 — Page 587
Source 3 — Page 586
```

 Users can expand individual sources to inspect the retrieved content.

 This provides greater transparency between:

```
Question
   ↓
Retrieved Evidence
   ↓
Generated Answer
```

 and allows users to inspect the material that was supplied to the language model.

---

 ## 🧪 Example

 ### Question

 > **What treatments are used for laryngeal cancer?**

 ### Retrieved Context

 The vector database may retrieve information containing passages such as:

```
Radiotherapy is the first choice to treat small lesions...

If the cancer recurs later, surgery may be attempted...

Laser excision surgery is used...

A combination of surgery and radiation therapy is often used...
```

 ### Generated Answer

```
According to the provided documents, treatments for laryngeal
cancer include radiotherapy, surgery, laser excision surgery,
and combinations of surgery and radiation therapy.
```

 The application also displays the relevant source documents and page numbers.

 > **Important:** The exact answer depends on the documents retrieved by the vector database and the model's interpretation of that context.

---

 ## 🚫 Out-of-Context Questions

 MedicalBot is designed to avoid answering questions when the required information cannot be found in the document collection.

 For example:

 ### Question

```
What is the capital of France?
```

 If the medical document collection contains no relevant information, the intended response is:

```
Answer not present in provided docs.
```

 This behavior is an important part of the document-grounding strategy.

 ### Intended Flow

```
Question
   │
   ▼
FAISS Retrieval
   │
   ▼
Relevant Medical Context?
   │
   ├── Yes ──► Generate grounded answer
   │
   └── No ───► Answer not present in provided docs
```

 > **Note:** Retrieval-based filtering is not a guarantee that the model will never generate unsupported information. Production systems should evaluate and monitor this behavior.

---

 ## 🔐 Security

 Never commit sensitive credentials to GitHub.

 Your `.gitignore` should include:

```
venv/
.env
__pycache__/
*.pyc
```

 If you use a `.env` file, it may contain:

```
HF_TOKEN=your_token_here
```

 Make sure `.env` is included in `.gitignore` before pushing the repository.

 ### 🔒 Recommended Security Practices

 - Never hard-code API tokens.
- Never commit `.env` files.
- Never publish private credentials.
- Rotate a token immediately if it is accidentally exposed.
- Use environment variables or your deployment platform's secret-management system.

---

 ## 📌 Current Pipeline

 The complete application works approximately like this:

```
                    ┌─────────────────┐
                    │      User       │
                    │    Question     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Streamlit    │
                    │    Chat UI      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Embedding    │
                    │     Model       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      FAISS      │
                    │  Vector Search  │
                    └────────┬────────┘
                             │
                       Top-K Documents
                             │
                             ▼
                    ┌─────────────────┐
                    │   RAG Prompt    │
                    │ Context + Query │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Hugging Face    │
                    │     Router      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    DeepSeek     │
                    │      LLM        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer + Sources│
                    └─────────────────┘
```

---

 ## 🎯 Project Goal

 The main goal of MedicalBot is to demonstrate how a **document-grounded conversational AI system** can be built using Retrieval-Augmented Generation.

 Instead of relying exclusively on an LLM's pre-trained knowledge, the application:

 1. 📚 Stores document information as vector embeddings.
2. 🔎 Retrieves relevant information using semantic search.
3. 🧩 Places the retrieved information into an LLM prompt.
4. 🤖 Generates an answer using the retrieved context.
5. 📄 Displays supporting source documents.
6. 🔍 Allows users to inspect the retrieved evidence.

 This approach makes the relationship between the:

```
User Question
      ↓
Retrieved Evidence
      ↓
Generated Response
```

 more transparent.

---

 ## 🚧 Future Improvements

 Several improvements could make MedicalBot more robust:

 - ✂️ Better document chunking strategies
- 🔎 Hybrid keyword + semantic search
- 🏆 Document re-ranking
- 🧠 Conversation memory
- ⚡ Streaming LLM responses
- 📚 Improved source citations
- 🔐 User authentication
- 🚫 Better handling of unsupported questions
- 📊 Retrieval accuracy evaluation
- 🧪 Automated RAG evaluation
- 🏥 More robust medical-document filtering
- 📈 Retrieval and answer-quality monitoring
- 📝 Structured citations for retrieved passages
- 🔄 Automatic document ingestion pipeline
- 💾 Persistent chat/session management

---

 ## 📊 Limitations

 Although the system attempts to restrict answers to the supplied documents, RAG does not automatically guarantee factual correctness.

 Potential failure points include:

```
Document Quality
      ↓
Chunking Quality
      ↓
Embedding Quality
      ↓
Retrieval Quality
      ↓
Prompt Quality
      ↓
LLM Generation
      ↓
Final Answer
```

 An error at any stage can affect the final response.

 Therefore, retrieved sources should be inspected when accuracy is important.

---

 ## ⚠️ Medical Disclaimer

 > **Important:** MedicalBot is an educational and research project. It is **not a substitute for professional medical advice, diagnosis, or treatment.**

 The system retrieves information from a document collection and generates answers based on that material.

 Like other RAG and LLM-based systems, it may produce:

 - ❌ Incorrect interpretations
- ❌ Incomplete information
- ❌ Outdated medical information
- ❌ Retrieval errors
- ❌ LLM-generated errors
- ❌ Unsupported or misleading responses

 Do not use MedicalBot as the sole basis for medical decisions.

 For medical concerns, diagnosis, treatment decisions, or emergencies, consult a qualified healthcare professional.

---

 ## 📦 Requirements

 A typical `requirements.txt` may contain:

```
langchain
langchain-community
langchain-core
langchain-huggingface
faiss-cpu
sentence-transformers
transformers
requests
streamlit
```

 Depending on the exact implementation, additional packages may be required.

---
