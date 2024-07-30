from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
TEMP_DIR = 'temp_files'
QUERY = """Extract the following details from the invoice:
- Invoice Type(PO/SO - Purchase Order or Sales Order)
- Invoice Number
- Invoice Date
- Seller Details
- Client/Purchaser Details
- Due Date
- Vendor Name
- Total Amount
- Line Items (description, quantity, unit price, total price) for all the items given in the invoice

Provide the extracted details in the following JSON format:
{   
    "invoice_type": "Purchase order",
    "invoice_number": "12345",
    "invoice_date": "2024-07-26",
    "due_date": "2024-08-26",
    "seller_details": [
        {
            "seller_name": "Inc.in",
            "address": "ABC",
            "contact": 847493,
            "tax_id": "4649530 X4583"
        }
    ],
    "client_details": [
        {
            "client_name": "ok.ihd",
            "address": "ABC",
            "contact": 847493,
            "tax_id": "4649530 X4583"
        }
    ],
    "total_amount": "1000.00",
    "line_items": [
        {
            "description": "Item 1",
            "quantity": "2",
            "unit_price": "250.00",
            "total_price": "500.00"
        },
        {
            "description": "Item 2",
            "quantity": "1",
            "unit_price": "500.00",
            "total_price": "500.00"
        }
    ]
}
Extract all the details accurately from the invoice and ensure that no values are missed."""

# Initialize embeddings model
def init_embeddings():
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Initialize Qdrant client
def init_qdrant_client(url):
    return QdrantClient(url=url, prefer_grpc=False)

# Initialize QA chain
def init_qa_chain():
    prompt_template = """Consider yourself an invoice extraction expert. Provide the best possible extraction of the details 
    from the given context. Don't give any explanation. Give only extracted details.

    context: {context}
    question: {question}
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Save uploaded file to temporary directory
def save_temp_file(file: UploadFile):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as temp_file:
        temp_file.write(file.file.read())
    return file_path

# Load and split PDF documents
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Save documents in Qdrant vector store
def save_in_qdrant(texts, collection_name, embeddings, url):
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name
    )

# Perform similarity search in Qdrant
def similarity_search(query, collection_name, embeddings, url):
    db = QdrantVectorStore(client=qdrant_client, embedding=embeddings, collection_name=collection_name)
    return db.similarity_search_with_score(query=query, k=5)

# Invoke the QA chain and get the response
def invoke_qa_chain(docs, query):
    response = qa_chain.invoke(
        {"input_documents": [doc for doc, score in docs], "question": query},
        return_only_outputs=True
    )
    return response["output_text"]

# Parse JSON response from the QA chain output
def parse_json_response(response_text):
    json_start_index = response_text.find("{")
    json_end_index = response_text.rfind("}")
    if json_start_index != -1 and json_end_index != -1:
        return json.loads(response_text[json_start_index:json_end_index + 1])
    return {"error": "Failed to parse the response"}

# Clean up temporary file
def cleanup_temp_file(file_path):
    os.remove(file_path)

# Initialize dependencies
embeddings = init_embeddings()
qdrant_client = init_qdrant_client(QDRANT_URL)
qa_chain = init_qa_chain()

@app.post("/extract_invoice/")
async def extract_invoice(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        temp_file_path = save_temp_file(file)

        # Load and split PDF documents
        texts = load_and_split_pdf(temp_file_path)

        # Save documents in Qdrant vector store
        collection_name = file.filename.replace('.', '_')
        save_in_qdrant(texts, collection_name, embeddings, QDRANT_URL)

        # Perform similarity search in Qdrant
        docs = similarity_search(QUERY, collection_name, embeddings, QDRANT_URL)

        # Invoke the QA chain and get the response
        response_text = invoke_qa_chain(docs, QUERY)

        # Parse the JSON response
        structured_response = parse_json_response(response_text)

        # Clean up the temporary file
        cleanup_temp_file(temp_file_path)

        return JSONResponse(content={"message": "Information extracted successfully", "response": structured_response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
