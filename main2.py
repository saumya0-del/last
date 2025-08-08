
# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests, tempfile, hashlib, threading, os

# NEW: env + openai-backed langchain
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# CHANGED: embeddings -> OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# =========================
# CONFIG
# =========================
BEARER_TOKEN = "ac96cb4db56939ddd84c8b78c7ac5eb9f288404f64af12fdf4d5aed51d1e3218"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Put it in .env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))

app = FastAPI(title="HackRx Intelligent Query–Retrieval API", version="1.3")

# =========================
# Health / Info (GET)
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "hint": "POST JSON to /api/v1/hackrx/run with Authorization: Bearer <token>"
    }

@app.get("/hackrx/run")
@app.get("/api/v1/hackrx/run")
def info():
    return {
        "message": "Use POST with Authorization: Bearer <token>",
        "schema": {"documents": "<url>", "questions": ["q1","q2","..."]}
    }

# =========================
# Auth (POST only)
# =========================
@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.method == "POST" and request.url.path.endswith("/hackrx/run"):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer Token")
        if auth.split(" ")[1] != BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid Token")
    return await call_next(request)

# =========================
# Schema
# =========================
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# =========================
# Utils: download + retriever cache
# =========================
def _download_pdf(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(tmp.name, "wb") as f:
        f.write(r.content)
    return tmp.name

_RETRIEVER_CACHE: Dict[str, any] = {}

def _build_retriever_from_url(pdf_url: str):
    key = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    path = _download_pdf(pdf_url)
    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=160)
    chunks = splitter.split_documents(docs)

    # CHANGED: OpenAI embeddings (offload heavy compute)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    _RETRIEVER_CACHE[key] = retriever
    return retriever

# =========================
# Lazy LLM (loads on first use) — now OpenAI
# =========================
_LLM = None
_LLM_LOCK = threading.Lock()

def _load_llm():
    global _LLM
    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM
        # CHANGED: Use OpenAI Chat model via LangChain
        _LLM = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            timeout=OPENAI_TIMEOUT,
            api_key=OPENAI_API_KEY,
        )
        return _LLM

# =========================
# Prompt + QA helper
# =========================
PROMPT = ChatPromptTemplate.from_template(
    "You are an expert in insurance policies. Answer ONLY from context. "
    "If not in context, say \"Not found in policy context.\" "
    "Context:\n{context}\n\nQuestion: {input}\nFinal Answer (one sentence):"
)

def _answer_batch(retriever, llm, questions: List[str]) -> List[str]:
    # document chain: stuffs retrieved docs into the prompt
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    # retrieval chain: handles retrieval + doc_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    out = []
    for q in questions:
        # key is "input" in v2 chains
        res = retrieval_chain.invoke({"input": q})
        # res["answer"] contains the generated answer
        out.append(res["answer"].strip())
    return out

# =========================
# Endpoint
# =========================
@app.post("/hackrx/run")
@app.post("/api/v1/hackrx/run")
def run_hackrx(req: HackRxRequest):
    retriever = _build_retriever_from_url(req.documents)
    llm = _load_llm()
    answers = _answer_batch(retriever, llm, req.questions)
    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
