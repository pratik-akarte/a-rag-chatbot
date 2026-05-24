import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_GENAI_API_KEY not found in .env")


def process_pdf(file_bytes: bytes, filename: str) -> tuple:
    """
    Takes raw PDF bytes, builds full RAG pipeline.
    Returns (retriever_chain, doc_info_dict)
    """

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip()]

    # Embed (one by one to avoid Gemini batching bug)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key,
        request_options={"timeout": 120}
    )

    texts = [d.page_content for d in docs]
    vectors = [embeddings.embed_query(t) for t in texts]

    # Vector Store
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=[
            {
                "source": d.metadata.get("source", filename),
                "page": d.metadata.get("page", 0)
            }
            for d in docs
        ]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = GoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

    # Prompt
    prompt = PromptTemplate(
        template="""Answer the question based only on the context below.
    If you don't know the answer, just say "I don't know" — don't make up an answer.
    
    Context:
    {context}
    
    Question:
    {input}
    
    Answer:""",
            input_variables=["context", "input"] )
    
    # Chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    os.unlink(tmp_path)

    doc_info = {
        "filename": filename,
        "pages": len(documents),
        "chunks": len(docs)
    }

    return retriever_chain, doc_info


def get_answer(retriever_chain, query: str) -> str:
    """
    Takes a built retriever_chain and a query string.
    Returns the LLM answer as a string.
    """
    response = retriever_chain.invoke({"input": query})
    return response["answer"]