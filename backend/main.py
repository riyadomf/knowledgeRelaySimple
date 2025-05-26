# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
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

# Modern LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.callbacks import get_openai_callback
from langsmith import Client
from langsmith.run_helpers import traceable
from langchain.callbacks import LangChainTracer
from langchain.globals import set_debug
from langchain_core.tracers.context import tracing_v2_enabled

import docx2txt
import httpx
import tempfile
import shutil
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

# Configuration for different LLM providers
class LLMConfig:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openrouter")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

llm_config = LLMConfig()



# LangSmith Configuration
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "knowledge-relay-v1")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Initialize LangSmith client
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    
    langsmith_client = Client(api_key=LANGSMITH_API_KEY, api_url=LANGSMITH_ENDPOINT)
    print(f"✅ LangSmith initialized - Project: {LANGSMITH_PROJECT}")
else:
    print("⚠️  LangSmith not configured - add LANGSMITH_API_KEY to .env file")
    langsmith_client = None


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








# Enhanced LLM Factory with LangSmith tracing
class LLMFactory:
    @staticmethod
    def get_llm():
        """Get LLM instance with LangSmith tracing"""
        callbacks = []
        if LANGSMITH_API_KEY:
            callbacks.append(LangChainTracer(project_name=LANGSMITH_PROJECT))
        
        if llm_config.provider == "openrouter":
            return ChatOpenAI(
                model=llm_config.model_name,
                api_key=llm_config.openrouter_api_key,
                base_url=llm_config.openrouter_base_url,
                temperature=0.1,
                max_tokens=1000,
                callbacks=callbacks,
                metadata={"provider": "openrouter", "model": llm_config.model_name}
            )
        elif llm_config.provider == "ollama":
            return Ollama(
                model=llm_config.model_name,
                base_url=llm_config.ollama_base_url,
                temperature=0.1,
                callbacks=callbacks,
                metadata={"provider": "ollama", "model": llm_config.model_name}
            )
        else:
            return ChatOpenAI(
                model=llm_config.model_name,
                api_key=llm_config.openai_api_key,
                temperature=0.1,
                max_tokens=1000,
                callbacks=callbacks,
                metadata={"provider": "openai", "model": llm_config.model_name}
            )
    
    @staticmethod
    def get_embeddings():
        """Get embeddings with metadata"""
        if llm_config.provider == "ollama":
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=llm_config.ollama_base_url
            )
        else:
            return OpenAIEmbeddings(
                api_key=llm_config.openrouter_api_key if llm_config.provider == "openrouter" else llm_config.openai_api_key,
                base_url=llm_config.openrouter_base_url if llm_config.provider == "openrouter" else None,
                model="text-embedding-ada-002"
            )

# Modern prompt templates
SYSTEM_PROMPT = """You are a knowledgeable software engineering assistant specializing in project knowledge transfer.
Your role is to provide accurate, actionable answers based on the provided context from project documentation and team knowledge.

Guidelines:
1. Base your answers strictly on the provided context
2. If the context is insufficient, clearly state what information is missing
3. Provide specific, actionable guidance rather than generic advice
4. Include relevant code snippets, commands, or configuration details when available
5. Organize complex answers with clear sections or bullet points
6. If multiple approaches exist, explain the trade-offs

Context Quality Assessment:
- High confidence: Multiple relevant sources with detailed information
- Medium confidence: Some relevant information but gaps exist  
- Low confidence: Limited or tangential information available

Always indicate your confidence level and reasoning."""

# Create modern prompt templates
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context from project knowledge base:
{context}

Question: {question}

Instructions:
1. Analyze the provided context for relevance to the question
2. Assess information quality and completeness
3. Provide a comprehensive answer based on available context
4. Clearly state confidence level and any limitations
5. If context is insufficient, suggest specific areas where more information is needed

Answer:""")
])

adaptive_questions_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at knowledge transfer. Generate specific, technical follow-up questions that would help capture important project knowledge that wasn't covered in the initial responses. Focus on practical details, edge cases, and operational knowledge."),
    ("human", "Based on the following responses, generate 3-5 relevant follow-up questions:\n\n{context}")
])













# Enhanced Knowledge Processor with tracing
class ModernKnowledgeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = LLMFactory.get_embeddings()
    
    @traceable(name="document_processing")
    async def process_document(self, file_path: str, file_type: str) -> List[Document]:
        """Process uploaded documents with tracing"""
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
            elif file_type == "md":
                text = UnstructuredMarkdownLoader.process(file_path)
                documents = [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Log metrics to LangSmith
        if langsmith_client:
            try:
                langsmith_client.create_run(
                    name="document_processing",
                    run_type="tool",
                    inputs={"file_path": file_path, "file_type": file_type},
                    outputs={
                        "chunks_created": len(chunks),
                        "total_documents": len(documents),
                        "avg_chunk_size": sum(len(chunk.page_content) for chunk in chunks) // len(chunks) if chunks else 0
                    }
                )
            except Exception as e:
                print(f"LangSmith logging error: {e}")
        
        return chunks
    
    @traceable(name="knowledge_storage")
    async def store_knowledge(self, team_id: str, documents: List[Document], entry_id: str):
        """Store knowledge with tracing"""
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
            
            return {"stored_chunks": len(texts), "collection": collection_name}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing knowledge: {str(e)}")
    
    @traceable(name="knowledge_search")
    async def search_knowledge(self, team_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge with detailed tracing"""
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
                    similarity = 1 - results['distances'][0][i]
                    search_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": similarity
                    })
            
            # Log search metrics
            if langsmith_client:
                try:
                    avg_similarity = sum(r['similarity'] for r in search_results) / len(search_results) if search_results else 0
                    langsmith_client.create_run(
                        name="knowledge_search",
                        run_type="retriever",
                        inputs={"query": query, "team_id": team_id, "top_k": top_k},
                        outputs={
                            "results_count": len(search_results),
                            "avg_similarity": avg_similarity,
                            "max_similarity": max(r['similarity'] for r in search_results) if search_results else 0
                        }
                    )
                except Exception as e:
                    print(f"LangSmith logging error: {e}")
            
            return search_results
            
        except Exception as e:
            return []

processor = ModernKnowledgeProcessor()



def calculate_confidence_score(search_results: List[Dict]) -> tuple[float, str]:
    """Calculate confidence score with reasoning"""
    if not search_results:
        return 0.0, "No relevant information found"
    
    similarities = [r['similarity'] for r in search_results[:3]]
    avg_similarity = sum(similarities) / len(similarities)
    
    high_quality_results = sum(1 for s in similarities if s > 0.8)
    medium_quality_results = sum(1 for s in similarities if 0.6 <= s <= 0.8)
    
    if high_quality_results >= 2:
        confidence = min(0.95, avg_similarity)
        reasoning = f"High confidence: {high_quality_results} highly relevant sources found"
    elif high_quality_results >= 1 or medium_quality_results >= 2:
        confidence = min(0.75, avg_similarity)
        reasoning = f"Medium confidence: Found {high_quality_results} highly relevant and {medium_quality_results} moderately relevant sources"
    else:
        confidence = min(0.5, avg_similarity)
        reasoning = "Low confidence: Limited relevant information available"
    
    return confidence, reasoning

def prepare_context_with_metadata(search_results: List[Dict]) -> str:
    """Prepare context with rich metadata for better LLM understanding"""
    if not search_results:
        return "No relevant context available."
    
    context_parts = []
    for i, result in enumerate(search_results[:3], 1):
        metadata = result['metadata']
        similarity = result['similarity']
        
        source_info = []
        if 'source_file' in metadata:
            source_info.append(f"File: {metadata['source_file']}")
        if 'type' in metadata:
            source_info.append(f"Type: {metadata['type']}")
        if 'developer' in metadata:
            source_info.append(f"Author: {metadata['developer']}")
        
        source_header = f"--- Source {i} (Relevance: {similarity:.2f}) ---"
        if source_info:
            source_header += f"\n[{', '.join(source_info)}]"
        
        context_parts.append(f"{source_header}\n{result['content']}\n")
    
    return "\n".join(context_parts)


# Enhanced chain creation with tracing
def create_qa_chain():
    """Create Q&A chain with enhanced tracing"""
    llm = LLMFactory.get_llm()
    
    def format_context(data):
        search_results = data["search_results"]
        return prepare_context_with_metadata(search_results)
    
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(format_context))
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "qa_chain"})
    
    return chain

def create_adaptive_questions_chain():
    """Create adaptive questions chain with tracing"""
    llm = LLMFactory.get_llm()
    
    def format_responses(data):
        responses = data["responses"]
        context = ""
        for question, answer in responses.items():
            context += f"Q: {question}\nA: {answer}\n\n"
        return context
    
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(format_responses))
        | adaptive_questions_prompt
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "adaptive_questions_chain"})
    
    return chain






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
    
    allowed_types = ['pdf', 'txt', 'docx', 'md']
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        documents = await processor.process_document(temp_file_path, file_extension)
        
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
        
        await processor.store_knowledge(team_id, documents, entry_id)
        
        return {"message": "Document uploaded and processed successfully", "entry_id": entry_id}
        
    finally:
        shutil.rmtree(temp_dir)

@app.get("/predefined-questions")
async def get_predefined_questions():
    """Get predefined questions for knowledge capture"""
    return {"questions": PREDEFINED_QUESTIONS}

@app.post("/adaptive-questions")
@traceable(name="adaptive_question_generation")
async def get_adaptive_questions(request: AdaptiveQuestionRequest):
    """Generate adaptive questions with tracing"""
    
    try:
        with tracing_v2_enabled(project_name=LANGSMITH_PROJECT) if LANGSMITH_API_KEY else nullcontext():
            chain = create_adaptive_questions_chain()
            
            result = await chain.ainvoke({
                "responses": request.previous_responses
            })
        
        questions = [q.strip() for q in result.split('\n') if q.strip() and ('?' in q)]
        
        # Log to LangSmith
        if langsmith_client:
            try:
                langsmith_client.create_run(
                    name="adaptive_questions",
                    run_type="chain",
                    inputs={"response_count": len(request.previous_responses)},
                    outputs={"questions_generated": len(questions)}
                )
            except Exception as e:
                print(f"LangSmith logging error: {e}")
        
        return {"adaptive_questions": questions[:5]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating adaptive questions: {str(e)}")


@app.post("/qa-session")
async def save_qa_session(request: QASessionRequest, db: Session = Depends(get_db)):
    """Save Q&A session responses"""
    
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
    
    await processor.store_knowledge(request.team_id, documents, entry_id)
    
    return {"message": "Q&A session saved successfully", "entry_id": entry_id}

@app.post("/ask", response_model=QuestionResponse)
@traceable(name="question_answering")
async def ask_question(request: QuestionRequest):
    """Answer question with comprehensive tracing"""
    
    try:
        # Search for relevant knowledge
        search_results = await processor.search_knowledge(request.team_id, request.question)
        
        # Calculate confidence
        confidence, confidence_reasoning = calculate_confidence_score(search_results)
        
        if not search_results:
            response = QuestionResponse(
                answer="I don't have enough information to answer this question. Please try rephrasing or ask the outgoing team member to provide more context.",
                sources=[],
                confidence=0.0
            )
            
            # Log failed query to LangSmith
            if langsmith_client:
                try:
                    langsmith_client.create_run(
                        name="failed_query",
                        run_type="chain",
                        inputs={"question": request.question, "team_id": request.team_id},
                        outputs={"reason": "no_relevant_knowledge", "confidence": 0.0},
                        error="No relevant knowledge found"
                    )
                except Exception as e:
                    print(f"LangSmith logging error: {e}")
            
            return response
        
        # Prepare sources
        sources = []
        for result in search_results[:3]:
            sources.append({
                "content": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                "metadata": result['metadata'],
                "similarity": result['similarity']
            })
        
        # Use tracing context for the chain
        with tracing_v2_enabled(project_name=LANGSMITH_PROJECT) if LANGSMITH_API_KEY else nullcontext():
            chain = create_qa_chain()
            
            answer_text = await chain.ainvoke({
                "search_results": search_results,
                "question": request.question
            })
        
        enhanced_answer = f"{answer_text}\n\n---\nConfidence Assessment: {confidence_reasoning}"
        
        response = QuestionResponse(
            answer=enhanced_answer,
            sources=sources,
            confidence=confidence
        )
        
        # Log successful query metrics
        if langsmith_client:
            try:
                langsmith_client.create_run(
                    name="successful_query",
                    run_type="chain",
                    inputs={
                        "question": request.question,
                        "team_id": request.team_id,
                        "sources_found": len(search_results)
                    },
                    outputs={
                        "confidence": confidence,
                        "answer_length": len(answer_text),
                        "sources_used": len(sources)
                    }
                )
            except Exception as e:
                print(f"LangSmith logging error: {e}")
        
        return response
        
    except Exception as e:
        # Log errors to LangSmith
        if langsmith_client:
            try:
                langsmith_client.create_run(
                    name="query_error",
                    run_type="chain",
                    inputs={"question": request.question, "team_id": request.team_id},
                    error=str(e)
                )
            except Exception as log_error:
                print(f"LangSmith error logging failed: {log_error}")
        
        if "rate limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        elif "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail="Authentication failed. Check API credentials.")
        else:
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/knowledge-entries/{team_id}")
async def get_knowledge_entries(team_id: str, db: Session = Depends(get_db)):
    """Get all knowledge entries for a team"""
    entries = db.query(KnowledgeEntry).filter(KnowledgeEntry.team_id == team_id).all()
    return entries

# Add health check endpoint with LangSmith status
@app.get("/health")
async def health_check():
    """Health check with LangSmith status"""
    langsmith_status = "enabled" if LANGSMITH_API_KEY else "disabled"
    return {
        "status": "healthy",
        "langsmith": langsmith_status,
        "project": LANGSMITH_PROJECT if LANGSMITH_API_KEY else None
    }

# Add context manager for non-async contexts
from contextlib import nullcontext

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)