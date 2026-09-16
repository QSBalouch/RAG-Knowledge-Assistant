import streamlit as st


from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import requests
from typing import Optional, List


# Hugging Face
HF_TOKEN = "YOUR TOKEN"
HF_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


# PAGE CONFIG
st.set_page_config(
    page_title="MedicalBot",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.chat-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 100px;
}

.user-message {
    display: flex;
    justify-content: flex-end;
}

.bot-message {
    display: flex;
    justify-content: flex-start;
}

.user-bubble {
    background-color: #0078ff;
    color: white;
    padding: 10px 15px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 16px;
}

.bot-bubble {
    background-color: #f1f1f1;
    color: black;
    padding: 10px 15px;
    border-radius: 18px 18px 18px 4px;
    max-width: 70%;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# Custom LLM
class HuggingFaceRouterLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "huggingface-router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": HF_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        response = requests.post(
            HF_ROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return answer

# Load FAISS

@st.cache_resource
def load_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vector_store/faiss_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# CREATE QA CHAIN

@st.cache_resource
def create_qa_chain():

    db = load_db()

    PROMPT_TEMPLATE = """
    You are a question-answering assistant.
    Answer the question using ONLY the information contained in the context.
    The context comes from medical documents.
    You may combine information from multiple parts of the context.
    If the context contains information that answers the question,
    give the answer using that information.
    If the context truly does not contain enough information,
    say exactly:
    Answer not present in provided docs.
    Do not use outside knowledge.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    llm = HuggingFaceRouterLLM()


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={
                "k": 3
            }
        ),
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )

    return qa_chain


# STREAMLIT UI
st.title("🩺 MedicalBot")
st.write(
    "Ask a medical question and get an answer based only on "
    "the information contained in the provided medical documents."
)

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# DISPLAY CHAT
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
                <div class="user-message">
                    <div class="user-bubble">
                        {message["content"]}
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
                <div class="bot-message">
                    <div class="bot-bubble">
                        {message["content"]}
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )

# LOAD MODEL
with st.spinner("Loading medical knowledge base..."):
    qa_chain = create_qa_chain()

# USER INPUT
user_query = st.chat_input("Ask a medical question...")

if user_query:

    # Add User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    # Display user message immediately
    st.markdown(
        f"""
            <div class="user-message">
                <div class="user-bubble">
                    {user_query}
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # GET RESPONSE FROM RAG

    with st.spinner("Searching Documents..."):
        reference = []
        try:
            response = qa_chain.invoke({
                "query": user_query
            })
            answer = response["result"].strip()
            reference = response["source_documents"]
        except Exception as e:
            answer = f"Error: {str(e)}"

    # SAVE BOT RESPONSE
    st.session_state.messages.append({
        "role": "bot",
        "content": answer 
    })

    st.markdown(
        f"""
            <div class="bot-message">
                <div class="bot-bubble">
                    {answer}
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # SOURCES

    if reference:
        st.subheader("Sources")
        for i, doc in enumerate(
            reference,
            start=1
        ):
            with st.expander(
                f"Source {i} - Page {doc.metadata.get('page', 'Unknown')}"
            ):
                st.write(doc.page_content)
                st.caption(
                    f"Source: {doc.metadata.get('source', 'Unknown')}"
                )
    
st.markdown('</div>', unsafe_allow_html=True)

