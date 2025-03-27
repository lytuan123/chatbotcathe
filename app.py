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
from fastapi.middleware.cors import CORSMiddleware # <-- THÊM IMPORT NÀY
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler

# --- Định nghĩa cấu trúc dữ liệu cho API ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- Class RAGPipeline (Tối ưu hóa và làm rõ) ---
class RAGPipeline:
    def __init__(self):
        """Khởi tạo pipeline RAG, bao gồm logging, OpenAI client, và load dữ liệu."""
        self._setup_logging()
        self.logger.info("Khởi tạo RAG Pipeline...")
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.critical("OPENAI_API_KEY không tìm thấy trong file .env")
                raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env")

            self.client = OpenAI(api_key=api_key)
            self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
            self.cache_file = Path(os.getenv("CACHE_FILE", "history_cache.json"))
            self.logger.info(f"Đường dẫn output: {self.output_dir.absolute()}")
            self.logger.info(f"Đường dẫn cache: {self.cache_file.absolute()}")

            self._initialize_pipeline_data()
            self._load_cache()
            self.logger.info("RAG Pipeline khởi tạo thành công.")

        except FileNotFoundError as fnf_error:
            self.logger.critical(f"Lỗi file cấu hình: {fnf_error}", exc_info=True)
            raise RuntimeError("Không thể khởi tạo RAG Pipeline do lỗi file cấu hình.") from fnf_error
        except ValueError as value_error:
            self.logger.critical(f"Lỗi giá trị cấu hình: {value_error}", exc_info=True)
            raise RuntimeError("Không thể khởi tạo RAG Pipeline do lỗi giá trị cấu hình.") from value_error
        except Exception as e:
            self.logger.critical(f"Lỗi không xác định khi khởi tạo RAGPipeline: {e}", exc_info=True)
            raise RuntimeError("Không thể khởi tạo RAG Pipeline do lỗi không xác định.") from e

    def _setup_logging(self):
        """Thiết lập cấu hình logging cho ứng dụng."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "rag_api.log" # Tên log file cụ thể hơn

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Đơn giản hóa format

        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8') # 10MB mỗi file
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        root_logger = logging.getLogger() # Logger gốc
        root_logger.setLevel(logging.INFO) # Mức INFO cho toàn bộ ứng dụng (có thể DEBUG cho dev)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

        self.logger = logging.getLogger(__name__) # Logger riêng cho class này
        self.logger.setLevel(logging.DEBUG) # Vẫn giữ DEBUG cho class RAGPipeline

    def _initialize_pipeline_data(self):
        """Load FAISS index và processed texts từ file."""
        index_path = self.output_dir / "faiss_index.bin"
        texts_path = self.output_dir / "processed_texts.pkl"

        if not self.output_dir.exists():
            raise FileNotFoundError(f"Thư mục output không tồn tại: {self.output_dir}")
        if not index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file index FAISS tại: {index_path}")
        if not texts_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file texts đã xử lý tại: {texts_path}")

        self.index = faiss.read_index(str(index_path))
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        if not isinstance(self.texts, list):
            raise ValueError("Dữ liệu texts đã xử lý phải là một list.")

        self.processed_texts = []
        for item in self.texts:
            if isinstance(item, dict):
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                page = metadata.get("page", "N/A")
                source = metadata.get("source", "N/A")
                self.processed_texts.append(f"[Trang {page} - {source}]\n{content}")
            else:
                self.processed_texts.append(str(item))
        self.logger.info(f"Đã load {len(self.processed_texts)} texts và FAISS index.")

    def _load_cache(self):
        """Load cache từ file JSON, tạo mới nếu không tồn tại."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                self.logger.info(f"Đã load cache với {len(self.cache)} mục từ: {self.cache_file}")
            else:
                self.cache = {}
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Đã tạo file cache mới tại: {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Lỗi khi load/tạo cache: {e}", exc_info=True)
            self.cache = {} # Vẫn khởi tạo cache rỗng để ứng dụng tiếp tục chạy

    def _save_cache(self, query: str, answer: str):
        """Lưu câu hỏi và câu trả lời vào cache file."""
        try:
            self.cache[query] = answer
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"Đã lưu cache cho query: {query[:50]}...")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cache: {e}", exc_info=True)

    def get_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding từ OpenAI API với cơ chế retry."""
        max_retries = 3
        delay = 1
        for attempt in range(max_retries):
            try:
                text_to_embed = text.replace("\n", " ").strip()
                if not text_to_embed:
                    self.logger.warning("Chuỗi rỗng được truyền vào get_embedding.")
                    raise ValueError("Không thể tạo embedding cho chuỗi rỗng.")

                response = self.client.embeddings.create(
                    input=[text_to_embed],
                    model="text-embedding-3-large"
                )
                if response.data and response.data[0].embedding:
                    return np.array(response.data[0].embedding, dtype=np.float32)
                else:
                    raise ValueError("API OpenAI không trả về embedding hợp lệ.")
            except Exception as e:
                self.logger.warning(f"Lỗi lấy embedding (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Không thể lấy embedding sau {max_retries} lần thử.")
                    raise # Ném lại lỗi cuối cùng
                time.sleep(delay * (attempt + 1))

        raise RuntimeError("Không thể lấy embedding sau nhiều lần thử lại.") # Lỗi cuối cùng nếu retry thất bại

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Tìm kiếm context liên quan nhất từ FAISS index."""
        try:
            query_embedding = self.get_embedding(query)
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k, self.index.ntotal)
            )
            threshold = 1.0 # Ngưỡng cần được điều chỉnh tùy theo dữ liệu và embedding model
            valid_indices = indices[0][distances[0] < threshold]

            if len(valid_indices) == 0:
                self.logger.info(f"Không tìm thấy context phù hợp cho query: {query[:50]}...")
                return "Không tìm thấy thông tin liên quan trong tài liệu."

            contexts = [self.processed_texts[i] for i in valid_indices if 0 <= i < len(self.processed_texts)]
            self.logger.info(f"Tìm thấy {len(contexts)} context phù hợp.")
            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            self.logger.error(f"Lỗi trong get_relevant_context: {e}", exc_info=True)
            return f"Lỗi khi tìm kiếm thông tin liên quan: {str(e)}" # Trả về thông báo lỗi thay vì raise

    def get_answer(self, query: str) -> str:
        """Trả lời câu hỏi dựa trên context và OpenAI API, sử dụng cache."""
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

        if query in self.cache:
            self.logger.info(f"Cache hit cho query: {query[:50]}...")
            return self.cache[query]
        self.logger.info(f"Cache miss cho query: {query[:50]}...")

        try:
            context = self.get_relevant_context(query)
            if context.startswith("Lỗi khi tìm kiếm thông tin liên quan:") or context == "Không tìm thấy thông tin liên quan trong tài liệu.":
                self.logger.warning(f"Context không phù hợp hoặc lỗi khi tìm kiếm: {context}")
                # Tiếp tục gọi LLM ngay cả khi context không tối ưu, để LLM tự quyết định

            prompt = f"""Bạn là một trợ lý AI chuyên nghiệp, chuyên trả lời câu hỏi về nghiệp vụ điều tra biến động dân số ngày 1/04/2025.
            Sử dụng thông tin trong phần "Context" dưới đây để trả lời câu hỏi một cách chính xác và chi tiết nhất.
            Nếu thông tin không có trong Context, hãy trả lời "Thông tin này không có trong tài liệu được cung cấp." và KHÔNG bịa đặt.
            Chỉ trả lời trực tiếp vào câu hỏi, không cần lời chào hay giới thiệu.

            Context:
            ---
            {context}
            ---

            Câu hỏi: {query}

            Trả lời:"""

            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            answer = response.choices[0].message.content.strip()
            end_time = time.time()
            self.logger.info(f"Gọi OpenAI thành công ({end_time - start_time:.2f}s).")

            self._save_cache(query, answer)
            return answer

        except HTTPException as http_error: # Bắt lại HTTPException từ đầu hàm
            raise http_error # Re-raise để FastAPI xử lý
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API OpenAI hoặc xử lý câu trả lời: {e}", exc_info=True)
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise HTTPException(status_code=429, detail="Hệ thống đang quá tải (OpenAI Rate Limit). Vui lòng thử lại sau.")
            elif "authentication" in error_msg.lower():
                raise HTTPException(status_code=401, detail="Lỗi xác thực OpenAI. Kiểm tra lại API key.")
            else:
                raise HTTPException(status_code=503, detail=f"Lỗi dịch vụ AI: {error_msg}")

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Population Survey RAG API",
    description="API trả lời câu hỏi về nghiệp vụ điều tra biến động dân số.",
    version="1.0.0"
)

# --- Cấu hình CORS Middleware --- # <-- THÊM PHẦN NÀY
origins = [
    "http://localhost",  # Cho phép từ localhost (nếu bạn test từ trình duyệt cục bộ)
    "http://localhost:8080", # Ví dụ cổng phát triển frontend/Android emulator
    "app://*",  # Một số lược đồ có thể được sử dụng bởi ứng dụng di động
    # "https://your-android-app-origin.com", # THÊM ORIGIN CỤ THỂ CỦA BẠN NẾU CÓ
    "*" # HOẶC CHO PHÉP TẤT CẢ ORIGINS (dùng thận trọng trong production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Danh sách các origin được phép
    allow_credentials=True, # Cho phép gửi cookie (nếu cần)
    allow_methods=["*"],    # Cho phép tất cả các phương thức (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Cho phép tất cả các header
)

# --- Middleware log request ---
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Middleware để log thông tin request và response."""
    start_time = time.time()
    logger = logging.getLogger("api.requests") # Logger riêng cho request logs
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request) # Gọi endpoint function

    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} (took {process_time:.4f}s)")
    return response

# --- Khởi tạo RAG Pipeline khi startup ---
try:
    rag_pipeline = RAGPipeline()
    logging.info("RAG Pipeline khởi tạo thành công.")
except RuntimeError as init_error:
    logging.critical(f"KHÔNG THỂ KHỞI ĐỘNG ỨNG DỤNG: Lỗi khởi tạo RAG Pipeline: {init_error}")
    # Xử lý dừng ứng dụng ở đây nếu cần thiết, ví dụ:
    # import sys
    # sys.exit(1) # Dừng ứng dụng với mã lỗi khác 0
    rag_pipeline = None # Đảm bảo biến rag_pipeline được gán giá trị ngay cả khi lỗi

# --- API Endpoints ---
@app.post("/answer", response_model=AnswerResponse, summary="Trả lời câu hỏi", description="API endpoint để trả lời câu hỏi về nghiệp vụ điều tra dân số.")
async def get_api_answer(request: QuestionRequest):
    """Endpoint API chính để nhận câu hỏi và trả về câu trả lời."""
    if rag_pipeline is None: # Kiểm tra lại pipeline đã khởi tạo thành công
        raise HTTPException(status_code=503, detail="Dịch vụ chưa sẵn sàng. RAG Pipeline không khởi tạo được.")
    try:
        answer = rag_pipeline.get_answer(request.question)
        return AnswerResponse(answer=answer)
    except HTTPException as e:
        rag_pipeline.logger.warning(f"Lỗi API xử lý '{request.question[:50]}...': {e.status_code} - {e.detail}")
        raise e # Re-raise HTTPException để FastAPI xử lý response lỗi
    except Exception as e:
        rag_pipeline.logger.error(f"Lỗi không xác định tại endpoint /answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ không xác định: {str(e)}")

@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
async def health_check():
    """Endpoint kiểm tra sức khỏe của API."""
    return {"status": "ok"}

# --- Lưu ý: Không cần if __name__ == "__main__": khi chạy với Uvicorn.
# Uvicorn sẽ import `app` từ file `app.py` và chạy nó.
