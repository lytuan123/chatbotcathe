import os
import logging
import json
import pickle
import numpy as np
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware

# Định nghĩa request/response models (giữ nguyên)
class QuestionRequest(BaseModel):
    query: str
    message: str = None

class AnswerResponse(BaseModel):
    answer: str

class RAGPipeline:
    def __init__(self, output_dir='output', cache_file='history_cache.json', max_history=5):
        """
        Khởi tạo RAG Pipeline với tối ưu hóa hiệu suất và chi phí.
        
        Args:
            output_dir (str): Thư mục chứa index và text
            cache_file (str): Đường dẫn file cache
            max_history (int): Số lượng tin nhắn tối đa trong lịch sử hội thoại
        """
        self.logger = self._setup_logging()
        self.logger.info("Bắt đầu khởi tạo RAG Pipeline...")

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.output_dir = Path(output_dir)
        self.cache_file = Path(cache_file)
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []  # Lưu lịch sử hội thoại
        
        self._validate_paths()
        self._load_index_and_texts()
        self._load_cache()
        
        self.logger.info("✅ RAG Pipeline khởi tạo thành công!")

    def _setup_logging(self):
        """Cấu hình logging (giữ nguyên)."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = RotatingFileHandler(log_dir / "rag_pipeline.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger = logging.getLogger('RAGPipeline')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _validate_paths(self):
        """Kiểm tra đường dẫn (giữ nguyên)."""
        paths_to_check = [self.output_dir / "faiss_index.bin", self.output_dir / "processed_texts.pkl"]
        for path in paths_to_check:
            if not path.exists():
                raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")

    def _load_index_and_texts(self):
        """Tải FAISS index và texts (tối ưu hóa)."""
        self.index = faiss.read_index(str(self.output_dir / "faiss_index.bin"))
        with open(self.output_dir / "processed_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
        
        self.processed_texts = []
        for item in self.texts:
            content = item.get('content', '')[:500] if isinstance(item, dict) else str(item)[:500]  # Giới hạn độ dài
            metadata = item.get('metadata', {}) if isinstance(item, dict) else {}
            page = metadata.get('page', 'N/A')
            source = metadata.get('source', 'N/A')
            self.processed_texts.append(f"[Trang {page} - {source}]\n{content}")
        
        self.logger.info(f"✅ Tải {len(self.processed_texts)} texts thành công")

    def _load_cache(self):
        """Tải hoặc khởi tạo cache (giữ nguyên)."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
            self.logger.info(f"✅ Tải {len(self.cache)} mục cache")
        except Exception as e:
            self.logger.warning(f"⚠️ Lỗi tải cache: {e}. Khởi tạo cache mới.")
            self.cache = {}

    def get_embedding(self, text: str, model="text-embedding-3-large"):  # Dùng model nhỏ hơn để tiết kiệm chi phí
        """Lấy embedding với retry mechanism."""
        try:
            response = self.client.embeddings.create(
                input=[text.replace("\n", " ").strip()[:1000]],  # Giới hạn input
                model=model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy embedding: {e}")
            raise

    def get_relevant_context(self, query: str, k: int = 2) -> str:  # Giảm k để tối ưu
        """Tìm context liên quan với ngưỡng chặt hơn."""
        try:
            query_embedding = self.get_embedding(query)
            distances, indices = self.index.search(np.array([query_embedding]), min(k, len(self.processed_texts)))
            threshold = 0.5  # Ngưỡng chặt hơn
            valid_indices = [i for i, d in zip(indices[0], distances[0]) if d < threshold]
            if not valid_indices:
                return "Không tìm thấy thông tin phù hợp."
            contexts = [self.processed_texts[i] for i in valid_indices]
            return "\n".join(contexts)  # Giảm ký tự phân cách
        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            raise

    def get_answer(self, query: str):
        """Xử lý trả lời với lịch sử hội thoại và tối ưu chi phí."""
        query = query.strip()
        if not query:
            return "Vui lòng nhập câu hỏi cụ thể."

        # Kiểm tra cache
        cache_key = f"{query}||{json.dumps(self.conversation_history[-2:])}"  # Cache dựa trên query và 2 tin nhắn cuối
        if cache_key in self.cache:
            self.logger.info("✅ Trả lời từ cache")
            return self.cache[cache_key]

        # Xử lý câu hỏi nối tiếp
        if "còn" in query.lower() and self.conversation_history:
            previous_query = self.conversation_history[-1]["content"]
            query = f"{previous_query} {query}"

        try:
            context = self.get_relevant_context(query)
            # Prompt ngắn gọn hơn
            system_prompt = "Bạn là trợ lý AI chuyên về điều tra dân số. Trả lời ngắn gọn, chính xác dựa trên context."
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history[-self.max_history:],  # Chỉ lấy max_history tin nhắn cuối
                {"role": "user", "content": f"Context: {context}\nCâu hỏi: {query}"}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",  # Dùng model nhỏ hơn, rẻ hơn
                messages=messages,
                temperature=0.3,
                max_tokens=500  # Giới hạn output
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Cập nhật lịch sử
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            if len(self.conversation_history) > self.max_history * 2:  # Giới hạn kích thước lịch sử
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            # Lưu cache
            self.cache[cache_key] = answer
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            return answer
        
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý câu hỏi: {e}")
            return "Xin lỗi, đã có lỗi xảy ra."

# FastAPI setup (giữ nguyên logic request body)
app = FastAPI(
    title="Population Survey RAG API",
    description="API trả lời thông minh về điều tra dân số",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiwiki.vn"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Middleware log request."""
    start_time = time.time()
    logger = logging.getLogger("api.requests")
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} (took {process_time:.4f}s)")
    return response

@app.post("/answer", response_model=AnswerResponse)
async def get_api_answer(request: QuestionRequest):
    """Endpoint trả lời câu hỏi."""
    try:
        message_content = request.query or request.message
        if not message_content:
            raise HTTPException(status_code=400, detail="Yêu cầu phải có nội dung.")
        
        rag_pipeline = RAGPipeline()  # Khởi tạo pipeline
        answer = rag_pipeline.get_answer(message_content)
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái dịch vụ."""
    return {"status": "healthy"}
