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

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from concurrent.futures import ThreadPoolExecutor
from asyncio import get_event_loop
from fastapi.middleware.gzip import GZipMiddleware
from collections import defaultdict
import asyncio

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
        """Lấy embedding với caching và retry logic"""
        # Kiểm tra cache
        cached = embedding_cache.get(text)
        if cached is not None:
            return cached
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=model
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                # Lưu vào cache
                embedding_cache.set(text, embedding)
                return embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} for embedding")
                time.sleep(1)

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Lấy context với tối ưu hóa cao hơn và xử lý lỗi chỉ số"""
        try:
            # Tiền xử lý query để tăng chất lượng tìm kiếm
            query = query.strip().lower()
            
            # Lấy embedding
            query_embedding = self.get_embedding(query)

            # Tìm kiếm và giới hạn số lượng kết quả
            max_results = min(k * 3, len(self.processed_texts))
            distances, indices = self.index.search(
                np.array([query_embedding]),
                max_results
            )

            # Chuyển sang numpy array để dễ xử lý
            distances_array = distances[0]
            indices_array = indices[0]

            # Áp dụng ngưỡng động
            if len(distances_array) > 0:
                mean_dist = np.mean(distances_array)
                std_dist = np.std(distances_array)
                dynamic_threshold = min(0.8, mean_dist + 1.5 * std_dist)
            else:
                dynamic_threshold = 0.75
            
            # Lọc các kết quả dưới ngưỡng - đảm bảo chỉ số trong phạm vi
            valid_pairs = [(idx, dist) for idx, dist in zip(indices_array, distances_array) 
                          if dist < dynamic_threshold]
            
            if not valid_pairs:
                return "Không tìm thấy context phù hợp."

            # Sắp xếp theo khoảng cách (giá trị nhỏ hơn = tương đồng hơn)
            sorted_results = sorted(valid_pairs, key=lambda x: x[1])
            
            # Nhóm theo nguồn - sửa cách xử lý nguồn
            sources = {}
            for idx, score in sorted_results:
                text = self.processed_texts[idx]
                # Tìm nguồn từ định dạng của text
                try:
                    source = text.split("]")[0].replace("[", "").strip()
                    if "Nguồn:" in source:
                        source = source.split("|")[0].strip()
                    else:
                        source = "Không rõ nguồn"
                except:
                    source = "Không rõ nguồn"
                    
                if source not in sources:
                    sources[source] = []
                if len(sources[source]) < 2:  # Tối đa 2 đoạn từ mỗi nguồn
                    sources[source].append((idx, score))
            
            # Lấy đủ k kết quả từ các nguồn đa dạng
            diverse_results = []
            for source_results in sources.values():
                diverse_results.extend(source_results)
            # Sắp xếp và giới hạn số lượng
            diverse_results = sorted(diverse_results, key=lambda x: x[1])[:k]
            
            # Định dạng kết quả
            formatted_contexts = []
            for i, (idx, score) in enumerate(diverse_results, 1):
                similarity_percent = 100 * (1 - score)
                formatted_context = f"[Kết quả #{i} - Độ phù hợp: {similarity_percent:.1f}%]\n{self.processed_texts[idx]}"
                formatted_contexts.append(formatted_context)
            
            return "\n\n---\n\n".join(formatted_contexts)

        except Exception as e:
            self.logger.error(f"❌ Lỗi trong get_relevant_context: {str(e)}", exc_info=True)
            # Trả về thông báo lỗi chi tiết hơn để gỡ lỗi
            if "index" in str(e) and "out of bounds" in str(e):
                return f"Lỗi chỉ số mảng: {str(e)}. Vui lòng liên hệ quản trị viên."
            return "Đã xảy ra lỗi khi tìm kiếm thông tin. Vui lòng thử lại sau."

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
            Hãy trả lời dựa trên context được cung cấp dưới đây một cách súc tích, chính xác và có cấu trúc.
            
            Quy tắc trả lời:
            1. Phân tích context kỹ lưỡng để tìm thông tin liên quan nhất đến câu hỏi.
            2. Trả lời súc tích, đầy đủ, rõ ràng và dễ hiểu, phân đoạn hợp lý.
            3. Nếu nhiều nguồn thông tin mâu thuẫn, hãy so sánh và giải thích sự khác biệt.
            4. Nêu rõ nguồn thông tin (trang, nguồn) khi trả lời.
            5. Nếu context không chứa đủ thông tin, hãy nói rõ và đưa ra gợi ý.
            6. Luôn liên kết với các câu hỏi và trả lời trước đó nếu có liên quan.
            7.Hãy là 1 trợ lý thân thiện, biết cách giao tiếp, chào hỏi với người dùng.
            
            Mục tiêu là cung cấp câu trả lời chất lượng cao, súc tích, thông tin chính xác và đầy đủ nhất có thể."""
            
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
                model="o3-mini",
                messages=messages,
               
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

# Thêm rate limiter tùy chỉnh
class RateLimiter:
    def __init__(self, limit=100, window=60):
        self.limit = limit  # Số request tối đa
        self.window = window  # Khoảng thời gian (giây)
        self.requests = defaultdict(list)  # IP -> [timestamps]
        
    def is_rate_limited(self, ip: str) -> bool:
        now = time.time()
        # Xóa timestamps cũ
        self.requests[ip] = [ts for ts in self.requests[ip] if now - ts < self.window]
        # Kiểm tra số lượng request
        if len(self.requests[ip]) >= self.limit:
            return True
        # Thêm timestamp mới
        self.requests[ip].append(now)
        return False

# Khởi tạo rate limiter
rate_limiter = RateLimiter(limit=100, window=60)

# Thêm middleware kiểm soát tốc độ
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if rate_limiter.is_rate_limited(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau."}
        )
    return await call_next(request)

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

# Thêm caching cho embeddings để giảm API calls
class EmbeddingCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, text):
        if text in self.cache:
            return self.cache[text]
        return None
        
    def set(self, text, embedding):
        # Giới hạn kích thước cache
        if len(self.cache) >= self.max_size:
            # Xóa một phần tử ngẫu nhiên
            self.cache.pop(next(iter(self.cache)))
        self.cache[text] = embedding

# Khởi tạo và tích hợp vào RAG Pipeline
embedding_cache = EmbeddingCache()

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
        # Lấy nội dung từ request
        message_content = request.query or request.message
        if not message_content:
            raise HTTPException(status_code=400, detail="Yêu cầu phải có nội dung.")
        
        # Kiểm tra trạng thái RAGPipeline
        global rag_pipeline
        if rag_pipeline is None:
            # Thử khởi tạo lại nếu cần
            try:
                rag_pipeline = RAGPipeline()
            except Exception as e:
                logging.error(f"Không thể khởi tạo RAGPipeline: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau."
                )
        
        # Giới hạn độ dài câu hỏi
        if len(message_content) > 500:
            message_content = message_content[:500]
            
        # Xử lý với timeout
        try:
            with ThreadPoolExecutor() as pool:
                answer = await get_event_loop().run_in_executor(
                    pool, rag_pipeline.get_answer, message_content
                )
                return AnswerResponse(answer=answer)
        except asyncio.TimeoutError:
            # Xử lý timeout
            logging.error("Timeout khi xử lý câu hỏi")
            raise HTTPException(
                status_code=504,
                detail="Xử lý câu hỏi mất quá nhiều thời gian. Vui lòng thử lại với câu hỏi ngắn hơn."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Lỗi xử lý câu hỏi: {str(e)}", exc_info=True)
        return AnswerResponse(
            answer="Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau hoặc liên hệ hỗ trợ kỹ thuật."
        )

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái dịch vụ."""
    return {"status": "healthy"}
