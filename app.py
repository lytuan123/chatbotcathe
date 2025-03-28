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
import hashlib

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
        sample_embedding = self.get_embedding("test", model="text-embedding-3-large")
        if self.index.d != len(sample_embedding):
            self.logger.warning("⚠️ Index không khớp với embedding. Tái tạo index...")
            self.rebuild_index()
             
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
        self.index = faiss.read_index(str(self.output_dir / "faiss_index.bin"))
        with open(self.output_dir / "processed_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
        
        self.processed_texts = []
        for item in self.texts:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            page = metadata.get('page', 'N/A')
            source = metadata.get('source', 'N/A')
            
            # Chia nhỏ văn bản thành các đoạn 1000 từ, có độ chồng chéo 100 từ
            chunks = self._chunk_text(content, chunk_size=1000, overlap=100)
            
            for idx, chunk in enumerate(chunks, 1):
                formatted_chunk = f"[Trang {page} - {source} - Đoạn {idx}]\n{chunk}"
                self.processed_texts.append(formatted_chunk)
        
        self.logger.info(f"✅ Tải {len(self.processed_texts)} text chunks thành công")
    
    def _chunk_text(self, text, chunk_size=1000, overlap=100):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

    def _load_cache(self):
        """Tải hoặc khởi tạo cache với xử lý lỗi an toàn"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                    
                # Kiểm tra và chuyển đổi cấu trúc cache nếu cần
                if not isinstance(loaded_cache, dict):
                    self.logger.warning("Cấu trúc cache không hợp lệ. Khởi tạo cache mới.")
                    self.cache = {}
                else:
                    # Chuyển đổi cache cũ sang cấu trúc mới nếu cần
                    self.cache = {}
                    for key, value in loaded_cache.items():
                        # Nếu value không phải dict, thêm timestamp
                        if not isinstance(value, dict):
                            self.cache[key] = {
                                'answer': value,
                                'timestamp': time.time()
                            }
                        else:
                            self.cache[key] = value
            else:
                self.cache = {}
            
            # Làm mới cache, loại bỏ các mục cũ
            current_time = time.time()
            self.cache = {
                k: v for k, v in self.cache.items() 
                if current_time - v.get('timestamp', current_time) < 7 * 24 * 3600
            }
            
            # Lưu lại cache đã được làm sạch
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Tải {len(self.cache)} mục cache")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Lỗi tải cache: {e}. Khởi tạo cache mới.")
            self.cache = {}
            
            # Tạo file cache trống nếu chưa tồn tại
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get_embedding(self, text: str, model="text-embedding-3-large"):  
        """Lấy embedding với retry mechanism."""
        try:
            response = self.client.embeddings.create(
                input=[text.replace("\n", " ").strip()[:2000]], 
                model=model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy embedding: {e}")
            raise

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        try:
            query_embedding = self.get_embedding(query, model="text-embedding-3-large")
            distances, indices = self.index.search(np.array([query_embedding]), min(k, len(self.processed_texts)))
            threshold = 0.7  
            valid_indices = [i for i, d in zip(indices[0], distances[0]) if d < threshold]
            if not valid_indices:
                return "Không tìm thấy thông tin phù hợp."
            contexts = [self.processed_texts[i] for i in valid_indices]
            return "\n\n---\n\n".join(contexts)
        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            raise

    def get_answer(self, query: str):
        query = query.strip()
        if not query:
            return "Vui lòng nhập câu hỏi cụ thể."

        # Kiểm tra cache
        cache_key = f"{query}||{json.dumps(self.conversation_history[-2:] if self.conversation_history else [])}"
        if cache_key in self.cache:
            self.logger.info("✅ Trả lời từ cache")
            return self.cache[cache_key]

        try:
            context = self.get_relevant_context(query)
            
            system_prompt = """Bạn là trợ lý AI chuyên về điều tra dân số 1/05/2024 tại Việt Nam. 
            Nhiệm vụ chính: 
            - Trả lời câu hỏi dựa trên context và lịch sử hội thoại một cách chính xác, chi tiết và logic
            - Luôn ưu tiên thông tin từ context trước kiến thức chung
            
            Chi tiết thực hiện:
            1. Phân tích context:
               • Đọc toàn bộ context một cách kỹ lưỡng
               • Xác định từng phần thông tin liên quan
               • Không bỏ qua bất kỳ chi tiết nào
            
            2. Xử lý ngữ cảnh:
               • Nếu context không đủ, giải thích rõ điều còn thiếu
               • Liên kết thông tin từ các phần context
               • Ưu tiên thông tin từ context trước kiến thức chung
            
            3. Yêu cầu trả lời:
               • Trả lời chính xác, súc tích
               • Nếu câu hỏi không có trong context, thừa nhận minh bạch
               • Cung cấp thông tin chi tiết từ context
               • LUÔN dẫn chiếu được nguồn thông tin từ context
            
            Lưu ý quan trọng: 1 :Tính minh bạch và chính xác là ưu tiên hàng đầu!
                              2 : Thay vì dùng từ "context", hãy sử dụng từ "nghiệp vụ điều tra " để chuyên nghiệp hơn """
            
            messages = [{"role": "system", "content": system_prompt}]
            # Chỉ thêm lịch sử nếu câu hỏi có vẻ nối tiếp
            if self.conversation_history and any(word in query.lower() for word in ["còn", "thế", "vậy", "lý do", "tại sao"]):
                messages.extend(self.conversation_history[-3:])  # Giới hạn 3 tin nhắn gần nhất
            messages.append({"role": "user", "content": f"Context: {context}\nCâu hỏi: {query}"})

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=2000  # Trung bình giữa 1000 và 2000
            )
            answer = response.choices[0].message.content.strip()

            # Cập nhật lịch sử
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            if len(self.conversation_history) > 10:  # Giới hạn 5 cặp Q&A
                self.conversation_history = self.conversation_history[-10:]

            # Lưu cache
            self.cache[cache_key] = answer
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

            return answer
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý câu hỏi: {e}")
            return "Xin lỗi, đã có lỗi xảy ra."

    # Hàm tái tạo index (đảm bảo dùng text-embedding-3-large)
    def rebuild_index(self):
        embeddings = [self.get_embedding(text, model="text-embedding-3-large") for text in self.processed_texts]
        embeddings = np.array(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]  # 3072 với text-embedding-3-large
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.output_dir / "faiss_index.bin"))
        self.logger.info("✅ Đã tái tạo FAISS index thành công")

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
