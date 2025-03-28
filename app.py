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
from concurrent.futures import ThreadPoolExecutor
from asyncio import get_event_loop
from fastapi.middleware.throttling import ThrottlingMiddleware
from fastapi.middleware.gzip import GZipMiddleware

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
        """Cải tiến tải index và texts với xử lý lỗi tốt hơn"""
        try:
            self.index = faiss.read_index(str(self.output_dir / "faiss_index.bin"))
            
            with open(self.output_dir / "processed_texts.pkl", "rb") as f:
                self.texts = pickle.load(f)
            
            self.processed_texts = []
            for item in self.texts:
                content = item.get('content', '')
                metadata = item.get('metadata', {})
                page = metadata.get('page', 'N/A')
                source = metadata.get('source', 'N/A')
                
                # Tối ưu kích thước đoạn văn
                chunks = self._chunk_text(content, chunk_size=800, overlap=150)
                
                for idx, chunk in enumerate(chunks, 1):
                    # Cải tiến định dạng để dễ đọc hơn
                    formatted_chunk = f"[Nguồn: {source} | Trang: {page} | Đoạn {idx}/{len(chunks)}]\n{chunk}"
                    self.processed_texts.append(formatted_chunk)
            
            self.logger.info(f"✅ Tải {len(self.processed_texts)} text chunks thành công")
            
            # Kiểm tra dimensionality của embeddings
            sample_embedding = self.get_embedding("test", model="text-embedding-3-large")
            if self.index.d != len(sample_embedding):
                self.logger.warning(f"⚠️ Kích thước index ({self.index.d}) không khớp với embedding ({len(sample_embedding)}). Tái tạo index...")
                self.rebuild_index()
        
        except Exception as e:
            self.logger.error(f"❌ Lỗi tải index và texts: {str(e)}", exc_info=True)
            raise

    def _chunk_text(self, text, chunk_size=800, overlap=150):
        """Cải tiến chunking văn bản với kích thước phù hợp hơn"""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]  # Trả về nguyên văn nếu đủ ngắn
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Đảm bảo chunk không quá ngắn
            if len(words[i:i + chunk_size]) < 200 and i > 0:
                continue
            
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

    def get_embedding(self, text: str, model="text-embedding-3-large") -> np.ndarray:
        """Lấy embedding với retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=model
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} for embedding")
                time.sleep(1)

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Lấy context với similarity threshold tối ưu hơn"""
        try:
            query_embedding = self.get_embedding(query)

            # Tăng số lượng kết quả tìm kiếm ban đầu để lọc tốt hơn
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k * 2, len(self.processed_texts))  # Tìm nhiều hơn để có thể lọc
            )

            # Điều chỉnh ngưỡng similarity để phù hợp hơn
            threshold = 0.75  # Điều chỉnh ngưỡng similarity để có kết quả tốt hơn
            valid_indices = [i for i, d in zip(indices[0], distances[0])
                           if d < threshold]

            if not valid_indices:
                return "Không tìm thấy context phù hợp."

            # Sắp xếp kết quả theo độ tương đồng và lấy top k
            sorted_results = sorted([(i, d) for i, d in zip(valid_indices, [distances[0][i] for i in valid_indices])], 
                                   key=lambda x: x[1])
            
            top_indices = [i for i, _ in sorted_results[:k]]
            contexts = [self.processed_texts[i] for i in top_indices]
            
            # Thêm chất lượng của mỗi kết quả vào context
            formatted_contexts = []
            for idx, (i, score) in enumerate(sorted_results[:k], 1):
                formatted_context = f"[Kết quả #{idx} - Độ tương đồng: {100*(1-score):.1f}%]\n{self.processed_texts[i]}"
                formatted_contexts.append(formatted_context)
            
            return "\n\n---\n\n".join(formatted_contexts)

        except Exception as e:
            self.logger.error(f"Lỗi trong get_relevant_context: {str(e)}", exc_info=True)
            raise

    def get_answer(self, query: str):
        """Xử lý trả lời với ngữ cảnh hội thoại thông minh."""
        query = query.strip()
        if not query:
            return "Vui lòng nhập câu hỏi cụ thể."

        # Kiểm tra cache với lịch sử gần nhất
        cache_key = f"{query}||{json.dumps(self.conversation_history[-3:])}"
        if cache_key in self.cache:
            self.logger.info("✅ Trả lời từ cache")
            return self.cache[cache_key]['answer']

        try:
            # Lấy context từ FAISS
            context = self.get_relevant_context(query, k=5)  # Tăng số lượng context

            # Cải thiện prompt chất lượng cao hơn
            system_prompt = """Bạn là trợ lý AI chuyên về điều tra dân số Việt Nam. 
            Hãy trả lời dựa trên context được cung cấp dưới đây một cách chi tiết, chính xác và có cấu trúc.
            
            Quy tắc trả lời:
            1. Phân tích context kỹ lưỡng để tìm thông tin liên quan nhất đến câu hỏi.
            2. Trả lời chi tiết, rõ ràng và dễ hiểu, phân đoạn hợp lý.
            3. Nếu nhiều nguồn thông tin mâu thuẫn, hãy so sánh và giải thích sự khác biệt.
            4. Nêu rõ nguồn thông tin (trang, nguồn) khi trả lời.
            5. Nếu context không chứa đủ thông tin, hãy nói rõ và đưa ra gợi ý.
            6. Luôn liên kết với các câu hỏi và trả lời trước đó nếu có liên quan.
            
            Mục tiêu là cung cấp câu trả lời chất lượng cao, thông tin chính xác và đầy đủ nhất có thể."""
            
            # Chuẩn bị context hiệu quả hơn
            context_prompt = f"CONTEXT ĐƯỢC CUNG CẤP:\n{context}\n\nLỊCH SỬ HỘI THOẠI GẦN ĐÂY:"
            
            # Thêm lịch sử hội thoại có định dạng rõ ràng
            history_text = ""
            for i in range(min(len(self.conversation_history), self.max_history * 2), 0, -2):
                if i-1 < 0:
                    break
                user_q = self.conversation_history[-i]["content"]
                assistant_a = self.conversation_history[-(i-1)]["content"]
                history_text += f"Người dùng: {user_q}\nTrợ lý: {assistant_a}\n\n"
            
            # Gọi GPT-4o với prompt cải tiến
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context_prompt}\n{history_text}Câu hỏi hiện tại: {query}\n\nTrả lời chi tiết:"}
            ]

            # Gọi GPT-4o với tham số tối ưu
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.2
            )
            answer = response.choices[0].message.content.strip()

            # Cập nhật lịch sử hội thoại
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            if len(self.conversation_history) > self.max_history * 3:
                self.conversation_history = self.conversation_history[-self.max_history * 3:]

            # Lưu cache với timestamp
            self.cache[cache_key] = {
                'answer': answer,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

            return answer

        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý câu hỏi: {str(e)}", exc_info=True)
            if "rate limit" in str(e).lower():
                return "Hệ thống đang bận, vui lòng thử lại sau ít phút."
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."

    # Hàm tái tạo index (đảm bảo dùng text-embedding-3-large)
    def rebuild_index(self):
        embeddings = [self.get_embedding(text, model="text-embedding-3-large") for text in self.processed_texts]
        embeddings = np.array(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]  # 3072 với text-embedding-3-large
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.output_dir / "faiss_index.bin"))
        self.logger.info("✅ Đã tái tạo FAISS index thành công")

# Thiết lập FastAPI app và CORS middleware
app = FastAPI(
    title="Population Survey RAG API",
    description="API trả lời thông minh về điều tra dân số",
    version="2.0.0"
)

# Cấu hình CORS để hỗ trợ cả website và ứng dụng Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiwiki.vn", "*"],  # Cho phép aiwiki.vn và tất cả origins cho Android
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],  # Cho phép các phương thức cần thiết
    allow_headers=["Content-Type", "Accept"],
)

# Thêm vào cấu hình app
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Nén các phản hồi lớn
app.add_middleware(
    ThrottlingMiddleware, 
    rate_limit=100,  # Số request tối đa mỗi phút
    rate_window=60,  # Thời gian tính rate limit (60 giây)
)

# Thêm logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Lấy thông tin request
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    # Xử lý request
    response = await call_next(request)
    
    # Tính thời gian xử lý
    process_time = time.time() - start_time
    
    # Log thông tin
    logging.info(f"IP: {client_ip} | Path: {path} | Status: {response.status_code} | Time: {process_time:.4f}s")
    
    return response

# Khởi tạo RAGPipeline một lần duy nhất khi server khởi động
try:
    rag_pipeline = RAGPipeline()
    logging.info("Khởi tạo RAGPipeline thành công")
except Exception as e:
    logging.critical(f"Lỗi khởi tạo RAGPipeline: {e}")
    rag_pipeline = None

@app.get("/")
async def root():
    """Endpoint gốc để kiểm tra trạng thái server."""
    return {"message": "API điều tra dân số đang hoạt động. Sử dụng /answer để đặt câu hỏi."}

@app.post("/answer", response_model=AnswerResponse)
async def get_api_answer(request: QuestionRequest):
    """Endpoint chính cho việc trả lời câu hỏi với timeout thích hợp"""
    try:
        # Lấy nội dung từ request - tương thích với cả query của Android
        message_content = request.query or request.message
        if not message_content:
            raise HTTPException(status_code=400, detail="Yêu cầu phải có nội dung.")
        
        # Giới hạn độ dài câu hỏi
        if len(message_content) > 500:
            message_content = message_content[:500]
            
        # Sử dụng thread pool để xử lý với timeout
        with ThreadPoolExecutor() as pool:
            answer = await get_event_loop().run_in_executor(
                pool, rag_pipeline.get_answer, message_content
            )
            
        return AnswerResponse(answer=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Lỗi xử lý câu hỏi: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái dịch vụ."""
    return {"status": "healthy"}
