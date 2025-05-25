# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import json
from datetime import datetime
import asyncio

# Database and Vector Store
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import chromadb
from chromadb.config import Settings

# LLM and Embeddings
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import docx2txt

# File processing
import PyPDF2
import tempfile
import shutil

# Environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="KnowledgeRelay API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./knowledge_relay.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Team(Base):
    __tablename__ = "teams"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"
    id = Column(String, primary_key=True)
    team_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_type = Column(String, nullable=False)  # document, qa_session, manual
    source_file = Column(String, nullable=True)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class TeamCreate(BaseModel):
    name: str

class TeamResponse(BaseModel):
    id: str
    name: str
    created_at: datetime

class QuestionRequest(BaseModel):
    team_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class QASessionRequest(BaseModel):
    team_id: str
    developer_name: str
    responses: Dict[str, str]

class AdaptiveQuestionRequest(BaseModel):
    team_id: str
    previous_responses: Dict[str, str]

# Predefined questions for knowledge capture
PREDEFINED_QUESTIONS = [
    "What is the overall architecture of this project?",
    "What are the key technologies and frameworks used?",
    "How do you deploy this application to production?",
    "What are the most common issues or bugs you encounter?",
    "What are the critical environment variables or configuration settings?",
    "How do you run tests and what testing strategies are used?",
    "What are the main dependencies and their purposes?",
    "What debugging tools and techniques do you use?",
    "What are the performance bottlenecks or optimization considerations?",
    "What documentation or resources are essential for new team members?"
]

class KnowledgeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    async def process_document(self, file_path: str, file_type: str) -> List[Document]:
        """Process uploaded documents and return chunks"""
        documents = []
        
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type == "txt":
                loader = TextLoader(file_path)
                documents = loader.load()
            elif file_type == "docx":
                text = docx2txt.process(file_path)
                documents = [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using OpenAI"""
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=texts
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

    async def store_knowledge(self, team_id: str, documents: List[Document], entry_id: str):
        """Store document chunks in ChromaDB"""
        collection_name = f"team_{team_id}"
        
        try:
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            texts = [doc.page_content for doc in documents]
            metadatas = [
                {
                    **doc.metadata,
                    "entry_id": entry_id,
                    "team_id": team_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for doc in documents
            ]
            ids = [f"{entry_id}_{i}" for i in range(len(documents))]
            
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing knowledge: {str(e)}")

    async def search_knowledge(self, team_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant knowledge chunks"""
        collection_name = f"team_{team_id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["distances", "metadatas", "documents"]
            )
            
            search_results = []
            if results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return search_results
            
        except Exception as e:
            return []

processor = KnowledgeProcessor()

# API Endpoints

@app.post("/teams", response_model=TeamResponse)
async def create_team(team: TeamCreate, db: Session = Depends(get_db)):
    """Create a new team"""
    team_id = str(uuid.uuid4())
    db_team = Team(id=team_id, name=team.name)
    db.add(db_team)
    db.commit()
    db.refresh(db_team)
    
    return TeamResponse(
        id=db_team.id,
        name=db_team.name,
        created_at=db_team.created_at
    )

@app.get("/teams", response_model=List[TeamResponse])
async def get_teams(db: Session = Depends(get_db)):
    """Get all teams"""
    teams = db.query(Team).all()
    return [TeamResponse(id=t.id, name=t.name, created_at=t.created_at) for t in teams]

@app.post("/upload-document")
async def upload_document(
    team_id: str = Form(...),
    developer_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    
    # Validate file type
    allowed_types = ['pdf', 'txt', 'docx']
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        documents = await processor.process_document(temp_file_path, file_extension)
        
        # Create knowledge entry
        entry_id = str(uuid.uuid4())
        db_entry = KnowledgeEntry(
            id=entry_id,
            team_id=team_id,
            title=file.filename,
            content=f"Document uploaded: {file.filename}",
            source_type="document",
            source_file=file.filename,
            created_by=developer_name
        )
        db.add(db_entry)
        db.commit()
        
        # Store in vector database
        await processor.store_knowledge(team_id, documents, entry_id)
        
        return {"message": "Document uploaded and processed successfully", "entry_id": entry_id}
        
    finally:
        # Cleanup temp file
        shutil.rmtree(temp_dir)

@app.get("/predefined-questions")
async def get_predefined_questions():
    """Get predefined questions for knowledge capture"""
    return {"questions": PREDEFINED_QUESTIONS}

@app.post("/adaptive-questions")
async def get_adaptive_questions(request: AdaptiveQuestionRequest):
    """Generate adaptive follow-up questions based on previous responses"""
    
    context = "Based on the following responses, generate 3-5 relevant follow-up questions:\n\n"
    for question, answer in request.previous_responses.items():
        context += f"Q: {question}\nA: {answer}\n\n"
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at knowledge transfer. Generate specific, technical follow-up questions that would help capture important project knowledge that wasn't covered in the initial responses. Focus on practical details, edge cases, and operational knowledge."
                },
                {"role": "user", "content": context}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        # Parse questions from response (assuming they're listed)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and ('?' in q)]
        
        return {"adaptive_questions": questions[:5]}  # Limit to 5 questions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating adaptive questions: {str(e)}")

@app.post("/qa-session")
async def save_qa_session(request: QASessionRequest, db: Session = Depends(get_db)):
    """Save Q&A session responses"""
    
    # Create knowledge entry
    entry_id = str(uuid.uuid4())
    content = json.dumps(request.responses, indent=2)
    
    db_entry = KnowledgeEntry(
        id=entry_id,
        team_id=request.team_id,
        title=f"Q&A Session - {request.developer_name}",
        content=content,
        source_type="qa_session",
        created_by=request.developer_name
    )
    db.add(db_entry)
    db.commit()
    
    # Convert Q&A to documents for vector storage
    documents = []
    for question, answer in request.responses.items():
        doc_content = f"Question: {question}\nAnswer: {answer}"
        documents.append(Document(
            page_content=doc_content,
            metadata={
                "type": "qa_pair",
                "question": question,
                "developer": request.developer_name
            }
        ))
    
    # Store in vector database
    await processor.store_knowledge(request.team_id, documents, entry_id)
    
    return {"message": "Q&A session saved successfully", "entry_id": entry_id}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question using the team's knowledge base"""
    
    # Search for relevant knowledge
    search_results = await processor.search_knowledge(request.team_id, request.question)
    
    if not search_results:
        return QuestionResponse(
            answer="I don't have enough information to answer this question. Please try rephrasing or ask the outgoing team member to provide more context.",
            sources=[],
            confidence=0.0
        )
    
    # Prepare context for LLM
    context = ""
    sources = []
    
    for result in search_results[:3]:  # Use top 3 results
        context += f"Source: {result['content']}\n\n"
        sources.append({
            "content": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
            "metadata": result['metadata'],
            "similarity": result['similarity']
        })
    
    # Generate answer using LLM
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about software projects based on provided context. Always provide specific, actionable answers. If the context doesn't contain enough information, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {request.question}\n\nPlease provide a comprehensive answer based on the context above."
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Calculate confidence based on search results
        avg_similarity = sum(r['similarity'] for r in search_results[:3]) / min(3, len(search_results))
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            confidence=avg_similarity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/knowledge-entries/{team_id}")
async def get_knowledge_entries(team_id: str, db: Session = Depends(get_db)):
    """Get all knowledge entries for a team"""
    entries = db.query(KnowledgeEntry).filter(KnowledgeEntry.team_id == team_id).all()
    return entries

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)