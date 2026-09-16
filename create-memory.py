from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the document
def load_document(data):
    loader = DirectoryLoader(
        path=data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    document = loader.load()  
    document = document[13:]
    return document

documents = load_document("data/")
print("Length",len(documents))

# Make Chunks
def create_chunks(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(doc)
    return chunks

chunks = create_chunks(documents)
print(len(chunks))

# Create Vector Embeddings
def create_vector_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embed_model = create_vector_embeddings()


# create Vector Store
VECTOR_STORE_PATH = "vector_store/faiss_db"

vector_store = FAISS.from_documents(
    chunks,
    embedding=embed_model,
)

vector_store.save_local(VECTOR_STORE_PATH)



