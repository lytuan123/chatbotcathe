import os
import logging
import json
import pickle
import numpy as np
from pathlib import Path
# BỎ: import gradio as gr # Không cần Gradio nữa
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import time

# THÊM: Import FastAPI và các thành phần cần thiết
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler # Di chuyển import này lên đây

# --- Định nghĩa cấu trúc dữ liệu cho API ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- Class RAGPipeline (Giữ nguyên gần như hoàn toàn) ---
class RAGPipeline:
    def __init__(self):
        """Khởi tạo pipeline với logging và cache từ tệp JSON."""
        self._setup_logging()
        try:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env")

            self.client = OpenAI(api_key=api_key)
            self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
            self.cache_file = Path(os.getenv("CACHE_FILE", "history_cache.json"))
            self.logger.info(f"Đường dẫn output: {self.output_dir.absolute()}")
            self.logger.info(f"Đường dẫn cache: {self.cache_file.absolute()}")

            self._initialize_pipeline()
            self._load_cache()
        except Exception as e:
            self.logger.error(f"Lỗi nghiêm trọng khi khởi tạo RAGPipeline: {str(e)}", exc_info=True)
            # Ném lỗi để ngăn ứng dụng khởi động nếu không thể khởi tạo pipeline
            raise RuntimeError(f"Không thể khởi tạo RAG Pipeline: {e}") from e

    def _setup_logging(self):
        """Thiết lập logging với file rotation."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler(
            log_dir / "rag_pipeline.log", maxBytes=1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Lấy logger gốc để cấu hình root logger (hoặc logger cụ thể nếu muốn)
        # Điều này đảm bảo các logger khác (ví dụ từ FastAPI/Uvicorn) cũng có thể được định dạng
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO) # Đặt level INFO cho root (hoặc DEBUG nếu cần log nhiều hơn)
        # Xóa các handler hiện có để tránh trùng lặp (nếu có)
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

        self.logger = logging.getLogger(__name__) # Logger riêng cho class này
        self.logger.setLevel(logging.DEBUG) # Đặt level DEBUG cho logger của class

    def _initialize_pipeline(self):
        """Khởi tạo FAISS index và load dữ liệu texts."""
        try:
            index_path = self.output_dir / "faiss_index.bin"
            texts_path = self.output_dir / "processed_texts.pkl"
            if not self.output_dir.exists(): raise FileNotFoundError(f"Thư mục output không tồn tại: {self.output_dir}")
            if not index_path.exists(): raise FileNotFoundError(f"Không tìm thấy file index tại {index_path}")
            if not texts_path.exists(): raise FileNotFoundError(f"Không tìm thấy file texts tại {texts_path}")

            self.index = faiss.read_index(str(index_path))
            with open(texts_path, "rb") as f: self.texts = pickle.load(f)
            if not isinstance(self.texts, list): raise ValueError("Processed texts phải là một list")

            self.processed_texts = []
            for item in self.texts:
                # Giữ nguyên logic xử lý texts
                if isinstance(item, dict):
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    page = metadata.get("page", "N/A")
                    source = metadata.get("source", "N/A")
                    self.processed_texts.append(f"[Trang {page} - {source}]\n{content}")
                else:
                    self.processed_texts.append(str(item))
            self.logger.info(f"Đã load {len(self.processed_texts)} texts và FAISS index")
        except Exception as e:
            self.logger.error("Lỗi trong _initialize_pipeline", exc_info=True)
            raise # Ném lại lỗi để báo hiệu khởi tạo thất bại

    def _load_cache(self):
        """Load cache từ tệp JSON hoặc tạo mới nếu không tồn tại."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f: self.cache = json.load(f)
                self.logger.info(f"Đã load cache với {len(self.cache)} mục từ {self.cache_file}")
            else:
                self.cache = {}
                with open(self.cache_file, 'w', encoding='utf-8') as f: json.dump(self.cache, f, ensure_ascii=False)
                self.logger.info(f"Đã tạo file cache mới tại {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Lỗi khi load/tạo cache: {str(e)}", exc_info=True)
            self.cache = {} # Khởi tạo cache rỗng nếu có lỗi

    def _save_cache(self, query: str, answer: str):
        """Lưu câu hỏi và câu trả lời vào cache và tệp JSON."""
        try:
            self.cache[query] = answer
            # Ghi đè toàn bộ file cache mỗi lần lưu
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"Đã lưu cache cho query: {query[:50]}...") # Log ngắn gọn
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cache: {str(e)}", exc_info=True)

    def get_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding từ OpenAI với retry logic."""
        max_retries = 3
        delay = 1 # Thời gian chờ giữa các lần retry (giây)
        for attempt in range(max_retries):
            try:
                text_to_embed = text.replace("\n", " ").strip() # Xử lý input cho embedding
                if not text_to_embed:
                    self.logger.warning("Chuỗi rỗng được truyền vào get_embedding.")
                    # Trả về vector zero hoặc ném lỗi tùy thuộc vào logic mong muốn
                    # Ở đây, ta ném lỗi để báo hiệu vấn đề
                    raise ValueError("Không thể tạo embedding cho chuỗi rỗng.")

                response = self.client.embeddings.create(
                    input=[text_to_embed], # Truyền vào dưới dạng list
                    model="text-embedding-3-large"
                )
                if response.data and response.data[0].embedding:
                    return np.array(response.data[0].embedding, dtype=np.float32)
                else:
                    raise ValueError("API OpenAI không trả về embedding hợp lệ.")
            except Exception as e:
                self.logger.warning(f"Lỗi khi lấy embedding (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Không thể lấy embedding sau {max_retries} lần thử.")
                    raise # Ném lại lỗi cuối cùng
                time.sleep(delay * (attempt + 1)) # Tăng thời gian chờ
        # Dòng này không bao giờ đạt được nếu raise ở trên hoạt động đúng
        raise RuntimeError("Không thể lấy embedding.")


    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Tìm context liên quan từ FAISS."""
        try:
            query_embedding = self.get_embedding(query)
            # Tìm kiếm trong FAISS index
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k, self.index.ntotal) # Đảm bảo k không lớn hơn số lượng vector trong index
            )

            # Lọc kết quả dựa trên ngưỡng khoảng cách (cần thử nghiệm để tìm giá trị phù hợp)
            # Ngưỡng này phụ thuộc vào không gian embedding và cách tính khoảng cách (thường là L2)
            threshold = 1.0 # Ví dụ: ngưỡng khoảng cách L2. Cần điều chỉnh!
            valid_indices = indices[0][distances[0] < threshold]

            if len(valid_indices) == 0:
                self.logger.info(f"Không tìm thấy context phù hợp cho query: {query[:50]}...")
                return "Không tìm thấy thông tin liên quan trong tài liệu."

            # Lấy context từ processed_texts
            contexts = [self.processed_texts[i] for i in valid_indices if 0 <= i < len(self.processed_texts)]
            self.logger.info(f"Tìm thấy {len(contexts)} context phù hợp.")
            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            # Trả về thông báo lỗi để endpoint API có thể xử lý
            return f"Lỗi khi tìm kiếm thông tin liên quan: {str(e)}"

    def get_answer(self, query: str) -> str:
        """
        Lấy câu trả lời cho query, sử dụng cache, context và OpenAI.
        Ném HTTPException nếu có lỗi có thể xử lý ở tầng API.
        """
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

        # 1. Kiểm tra cache
        if query in self.cache:
            self.logger.info(f"Cache hit cho query: {query[:50]}...")
            return self.cache[query]
        self.logger.info(f"Cache miss cho query: {query[:50]}...")

        # 2. Lấy context
        try:
            context = self.get_relevant_context(query)
            # Kiểm tra nếu context trả về thông báo lỗi
            if context.startswith("Lỗi khi tìm kiếm thông tin liên quan:") or \
               context == "Không tìm thấy thông tin liên quan trong tài liệu.":
                self.logger.warning(f"Context không phù hợp hoặc lỗi khi tìm kiếm: {context}")
                # Quyết định xử lý: Có thể vẫn hỏi LLM hoặc trả lỗi ngay
                # Ở đây ta vẫn tiếp tục để LLM xử lý
        except Exception as e:
            # Lỗi nghiêm trọng khi lấy context (ví dụ lỗi embedding)
            self.logger.error("Lỗi nghiêm trọng khi gọi get_relevant_context", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Lỗi nội bộ khi tìm kiếm context: {str(e)}")


        # 3. Tạo prompt
        prompt = f"""Bạn là một trợ lý AI chuyên nghiệp, được huấn luyện để trả lời các câu hỏi dựa trên tài liệu về nghiệp vụ điều tra biến động dân số ngày 1/04/2025.
        Hãy sử dụng thông tin trong phần "Context" dưới đây để trả lời câu hỏi một cách chính xác và chi tiết nhất có thể.
        Nếu thông tin không có trong Context, hãy nói rõ "Thông tin này không có trong tài liệu được cung cấp." và KHÔNG cố gắng bịa đặt câu trả lời.
        Chỉ trả lời trực tiếp vào câu hỏi, không thêm lời chào hay giới thiệu không cần thiết.

        Context:
        ---
        {context}
        ---

        Câu hỏi: {query}

        Trả lời:"""

        # 4. Gọi API OpenAI
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o", # Hoặc "gpt-3.5-turbo" nếu muốn tiết kiệm chi phí
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # Giảm temperature để câu trả lời bám sát context hơn
                max_tokens=1500 # Giới hạn độ dài trả về
            )
            end_time = time.time()
            answer = response.choices[0].message.content.strip()
            self.logger.info(f"Gọi OpenAI thành công ({end_time - start_time:.2f}s).")

            # 5. Lưu vào cache
            self._save_cache(query, answer)
            return answer

        except Exception as e:
            self.logger.error("Lỗi khi gọi API OpenAI", exc_info=True)
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise HTTPException(status_code=429, detail="Hệ thống đang quá tải (OpenAI Rate Limit). Vui lòng thử lại sau.")
            elif "authentication" in error_msg.lower():
                 raise HTTPException(status_code=401, detail="Lỗi xác thực OpenAI. Kiểm tra lại API key.")
            else:
                raise HTTPException(status_code=503, detail=f"Lỗi khi giao tiếp với dịch vụ AI: {error_msg}")


# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Population Survey RAG API",
    description="API trả lời câu hỏi về nghiệp vụ điều tra biến động dân số.",
    version="1.0.0"
)

# --- Middleware để log request (Tùy chọn) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # Log thông tin cơ bản về request
    logging.getLogger("api.requests").info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    # Log thông tin về response
    logging.getLogger("api.requests").info(f"Response: {response.status_code} (took {process_time:.4f}s)")
    return response


# --- Khởi tạo RAG Pipeline (một lần khi server khởi động) ---
# Đặt trong try-except để đảm bảo server không khởi động nếu pipeline lỗi
try:
    rag_pipeline = RAGPipeline()
    logging.info("RAG Pipeline đã được khởi tạo thành công.")
except Exception as e:
    logging.critical(f"KHÔNG THỂ KHỞI TẠO ỨNG DỤNG do lỗi RAG Pipeline: {e}", exc_info=True)
    # Thoát hoặc xử lý để ngăn Uvicorn chạy mà không có pipeline
    # Ví dụ: raise SystemExit(1) # Cách này sẽ dừng hoàn toàn
    # Hoặc để lỗi được ném ra từ __init__ xử lý (như đã làm ở trên)


# --- Định nghĩa API Endpoint ---
@app.post("/answer",
          response_model=AnswerResponse,
          summary="Trả lời câu hỏi",
          description="Nhận câu hỏi và trả về câu trả lời từ hệ thống RAG.")
async def get_api_answer(request: QuestionRequest):
    """
    Endpoint chính để xử lý yêu cầu trả lời câu hỏi.
    """
    if not rag_pipeline: # Kiểm tra lại (mặc dù đã có try-except ở trên)
         raise HTTPException(status_code=503, detail="Dịch vụ chưa sẵn sàng (Pipeline không khởi tạo được).")

    try:
        answer = rag_pipeline.get_answer(request.question)
        return AnswerResponse(answer=answer)
    except HTTPException as e:
        # Log lỗi đã được định dạng từ get_answer
        rag_pipeline.logger.warning(f"Lỗi API xử lý '{request.question[:50]}...': {e.status_code} - {e.detail}")
        # Ném lại để FastAPI trả về response lỗi chuẩn
        raise e
    except Exception as e:
        # Bắt các lỗi không mong muốn khác xảy ra ở tầng API
        rag_pipeline.logger.error(f"Lỗi không xác định tại endpoint /answer", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ không xác định: {str(e)}")

# --- Endpoint kiểm tra sức khỏe (Health Check) ---
@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
async def health_check():
    """
    Trả về trạng thái 'ok' nếu API đang chạy.
    """
    # Có thể thêm kiểm tra phức tạp hơn ở đây (ví dụ: kiểm tra kết nối OpenAI, FAISS)
    return {"status": "ok"}

# BỎ: Phần khởi chạy Gradio
# if __name__ == "__main__":
#     ...

# Lưu ý: Không cần `if __name__ == "__main__":` khi chạy với Uvicorn.
# Uvicorn sẽ import `app` từ file `app.py` và chạy nó.
