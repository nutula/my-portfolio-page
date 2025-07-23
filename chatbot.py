from langchain.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_personal_data(file_path):
    """
    Loads personal text data for RAG pipeline.
    Replace TextLoader with other loaders for PDF, Docx, etc.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def load_llm(model_name="Qwen/Qwen3-0.6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=700,
    do_sample=False,  
    temperature=0.3 
    )
    
    llm = HuggingFacePipeline(pipeline=generator)
    return llm

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    prompt_template = """You are a helpful assistant.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def prepare_rag_pipeline(personal_data_path):
    # Load data
    docs = load_personal_data(personal_data_path)
    #print(f"Loaded {len(docs)} documents.")
    
    # Split into chunks
    chunks = chunk_documents(docs)
    #print(f"Split into {len(chunks)} chunks.")
    
    # Create vector store
    vectorstore = create_vector_store(chunks)
    #print("Vector store created.")
    
    # Load LLM
    llm = load_llm()
    #print("LLM loaded.")
    
    # Build RAG pipeline
    qa_chain = build_rag_chain(vectorstore, llm)
    #print("RAG pipeline ready.")
    
    return qa_chain


if __name__ == "__main__":
    qa = prepare_rag_pipeline("mydata.txt")
    
    while True:
        query = input("\nAsk me anything: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa.run(query)
        # Extract the first answer after "Answer:" and remove duplicates
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip().split("\n")[0]
        print(f"{answer}")