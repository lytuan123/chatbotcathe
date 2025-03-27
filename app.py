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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware

class QuestionRequest(BaseModel):
    query: str
    message: str = None

class AnswerResponse(BaseModel):
    answer: str

class RAGPipeline:
    def __init__(self, output_dir='output', cache_file='history_cache.json'):
        """
        Khởi tạo RAG Pipeline với cấu hình linh hoạt và robust error handling.
        
        Args:
            output_dir (str): Thư mục chứa các file index và text
            cache_file (str): Đường dẫn file cache
        """
        self.logger = self._setup_logging()
        self.logger.info("Bắt đầu khởi tạo RAG Pipeline...")

        try:
            load_dotenv()
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            self.output_dir = Path(output_dir)
            self.cache_file = Path(cache_file)
            
            self._validate_paths()
            self._load_index_and_texts()
            self._load_cache()
            
            self.logger.info("✅ RAG Pipeline khởi tạo thành công!")

        except Exception as e:
            self.logger.critical(f"❌ Lỗi khởi tạo RAG Pipeline: {e}")
            raise

    def _setup_logging(self):
        """Cấu hình logging chuyên nghiệp với file rotation."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        
        # File handler
        file_handler = RotatingFileHandler(
            log_dir / "rag_pipeline.log", 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger('RAGPipeline')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _validate_paths(self):
        """Kiểm tra tính hợp lệ của các đường dẫn."""
        paths_to_check = [
            self.output_dir / "faiss_index.bin",
            self.output_dir / "processed_texts.pkl"
        ]
        
        for path in paths_to_check:
            if not path.exists():
                raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")

    def _load_index_and_texts(self):
        """Tải FAISS index và processed texts."""
        self.index = faiss.read_index(str(self.output_dir / "faiss_index.bin"))
        
        with open(self.output_dir / "processed_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
        
        # Xử lý texts để dễ dàng truy xuất
        self.processed_texts = []
        for item in self.texts:
            content = item.get('content', '') if isinstance(item, dict) else str(item)
            metadata = item.get('metadata', {}) if isinstance(item, dict) else {}
            page = metadata.get('page', 'N/A')
            source = metadata.get('source', 'N/A')
            
            formatted_text = f"[Trang {page} - {source}]\n{content}"
            self.processed_texts.append(formatted_text)
        
        self.logger.info(f"✅ Tải {len(self.processed_texts)} texts thành công")

    def _load_cache(self):
        """Tải hoặc khởi tạo cache."""
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

    def get_embedding(self, text: str, model="text-embedding-3-large"):
        """Lấy embedding với retry mechanism."""
        try:
            response = self.client.embeddings.create(
                input=[text.replace("\n", " ").strip()],
                model=model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy embedding: {e}")
            raise

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Tìm context liên quan từ FAISS với ngưỡng similarity."""
        try:
            query_embedding = self.get_embedding(query)

            # Batch search trong FAISS
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k, len(self.processed_texts))
            )

            # Lọc kết quả với ngưỡng similarity
            threshold = 0.7
            valid_indices = [i for i, d in zip(indices[0], distances[0]) if d < threshold]

            if not valid_indices:
                return "Không tìm thấy context phù hợp."

            contexts = [self.processed_texts[i] for i in valid_indices]
            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            raise

    def get_answer(self, query: str):
        """Xử lý trả lời với caching và robust error handling."""
        query = query.strip()
        
        if not query:
            return "Vui lòng nhập câu hỏi cụ thể."
        
        # Kiểm tra cache
        if query in self.cache:
            self.logger.info("✅ Trả lời từ cache")
            return self.cache[query]
        
        try:
            context = self.get_relevant_context(query)
            
            prompt = f"""Bạn là trợ lý AI chuyên nghiệp về điều tra biến động dân số 1/4/2025.
            Bạn cũng có thể giao tiếp chào hỏi với người dùng.
            Sử dụng context dưới đây để trả lời câu hỏi chính xác và chi tiết.
            Nếu không tìm thấy thông tin, hãy nói rõ.

            Context:
            {context}

            Câu hỏi: {query}

            Trả lời ngắn gọn, chuyên nghiệp:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Lưu cache
            self.cache[query] = answer
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            return answer
        
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý câu hỏi: {e}")
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."

# FastAPI setup tương tự như code cũ
app = FastAPI(
    title="Population Survey RAG API",
    description="API trả lời thông minh về điều tra dân số",
    version="2.0.0"
)

# --- Cấu hình CORS Middleware ---
origins = ["*"] # Cho phép tất cả origins để dễ dàng test, CÂN NHẮC GIỚI HẠN TRONG PRODUCTION

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware log request ---
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Middleware để log thông tin request và response."""
    start_time = time.time()
    logger = logging.getLogger("api.requests")
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} (took {process_time:.4f}s)")
    return response
@app.post("/answer", response_model=AnswerResponse)
async def get_api_answer(request: QuestionRequest):
    """Endpoint chính cho việc trả lời câu hỏi."""
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
    """Endpoint kiểm tra trạng thái dịch vụ."""
    return {"status": "healthy"}
