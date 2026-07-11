"""Backend pipeline for the QA bot.

Loads one or more PDFs, splits them, builds embeddings, and creates a
conversational QA chain that remembers previous questions in the chat.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain


def load_pdf(file_path: str):
    """Load a single PDF file and return LangChain documents."""

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at path: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid file type. Expected a .pdf file, got: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse PDF '{file_path}': {e}")

    if not documents:
        raise ValueError(f"No pages could be loaded from: {file_path}")

    total_text = "".join(doc.page_content.strip() for doc in documents)
    if not total_text:
        raise ValueError(f"The PDF '{file_path}' appears to be empty or has no extractable text.")

    return documents


def load_pdfs(file_paths):
    """Load multiple PDFs and combine all their pages into one list.

    Each page still keeps its own metadata, so we can tell later which
    PDF a chunk came from if needed.
    """

    if not file_paths:
        raise ValueError("No files provided to load.")

    all_documents = []
    for path in file_paths:
        all_documents.extend(load_pdf(path))

    return all_documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller text chunks."""

    if not documents:
        raise ValueError("No documents provided to split.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Splitting produced no chunks. The document may be empty.")

    return chunks


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a HuggingFace embedding model."""

    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")

    return embeddings


def create_vector_store(chunks, embedding_model):
    """Create an in-memory Chroma store from document chunks.

    It stays in memory so each session starts fresh with only the
    PDFs the user just uploaded.
    """

    if not chunks:
        raise ValueError("No chunks provided to store.")

    if embedding_model is None:
        raise ValueError("No embedding model provided.")

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create Chroma vector store: {e}")

    return vector_store


def create_retriever(vector_store, k: int = 3):
    """Create a retriever from the vector store."""

    if vector_store is None:
        raise ValueError("No vector store provided.")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever: {e}")

    return retriever


def create_qa_chain(retriever, model_name: str = "llama-3.1-8b-instant"):
    """Build a conversational QA chain with Groq and the retriever.

    Unlike a plain QA chain, this one accepts chat history, so it can
    understand follow-up questions like "what about its scope?".
    """

    if retriever is None:
        raise ValueError("No retriever provided.")

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable not found. "
            "Set it before running the app."
        )

    try:
        llm = ChatGroq(
            model=model_name,
            temperature=0,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build QA chain: {e}")

    return qa_chain


def build_qa_pipeline(file_paths):
    """Run the full QA pipeline for one or more PDFs.

    Accepts a single file path or a list of file paths.
    """

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    documents = load_pdfs(file_paths)
    chunks = split_documents(documents)
    embeddings = get_embedding_model()
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store)
    qa_chain = create_qa_chain(retriever)

    return qa_chain


if __name__ == "__main__":
    test_paths = ["sample.pdf"]  # add more paths here to test multiple PDFs
    try:
        qa_chain = build_qa_pipeline(test_paths)
        print("Full pipeline executed successfully. QA chain is ready.")

        chat_history = []

        first_question = "What is this document about?"
        response = qa_chain.invoke({"question": first_question, "chat_history": chat_history})
        print("Q1:", first_question)
        print("A1:", response["answer"])

        chat_history.append((first_question, response["answer"]))

        # Follow-up question to test chat history working
        second_question = "Can you tell me more about that?"
        response2 = qa_chain.invoke({"question": second_question, "chat_history": chat_history})
        print("Q2:", second_question)
        print("A2:", response2["answer"])
    except Exception as err:
        print(f"Pipeline error: {err}")
