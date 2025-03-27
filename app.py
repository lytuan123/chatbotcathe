import os
import logging
import json
import pickle
import numpy as np
from pathlib import Path
import gradio as gr
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import time

class RAGPipeline:
    def __init__(self):
        """Khởi tạo pipeline với logging và cache từ tệp JSON."""
        self._setup_logging()

        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env")

            # Khởi tạo OpenAI client
            self.client = OpenAI(
                api_key=api_key
            )

            # Thiết lập đường dẫn từ biến môi trường
            self.output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
            self.cache_file = Path(os.getenv("CACHE_FILE", "history_cache.json"))
            self.logger.info(f"Đường dẫn output: {self.output_dir.absolute()}")
            self.logger.info(f"Đường dẫn cache: {self.cache_file.absolute()}")

            # Khởi tạo pipeline
            self._initialize_pipeline()

            # Load cache từ tệp JSON
            self._load_cache()

        except Exception as e:
            self.logger.error(f"Lỗi trong __init__: {str(e)}", exc_info=True)
            raise

    def _setup_logging(self):
        """Thiết lập logging với file rotation."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / "rag_pipeline.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _initialize_pipeline(self):
        """Khởi tạo FAISS index và load dữ liệu texts."""
        try:
            index_path = self.output_dir / "faiss_index.bin"
            texts_path = self.output_dir / "processed_texts.pkl"

            if not self.output_dir.exists():
                raise FileNotFoundError(f"Thư mục output không tồn tại: {self.output_dir}")
            if not index_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file index tại {index_path}")
            if not texts_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file texts tại {texts_path}")

            self.index = faiss.read_index(str(index_path))
            with open(texts_path, "rb") as f:
                self.texts = pickle.load(f)

            if not isinstance(self.texts, list):
                raise ValueError("Processed texts phải là một list")

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

            self.logger.info(f"Đã load {len(self.processed_texts)} texts và FAISS index")

        except Exception as e:
            self.logger.error("Lỗi trong _initialize_pipeline", exc_info=True)
            raise

    def _load_cache(self):
        """Load cache từ tệp JSON hoặc tạo mới nếu không tồn tại."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False)
            self.logger.info(f"Đã load cache với {len(self.cache)} mục")
        except Exception as e:
            self.logger.error(f"Lỗi khi load cache: {str(e)}", exc_info=True)
            self.cache = {}

    def _save_cache(self, query: str, answer: str):
        """Lưu câu hỏi và câu trả lời vào cache và tệp JSON."""
        try:
            self.cache[query] = answer
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Đã lưu câu trả lời vào cache cho query: {query}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cache: {str(e)}", exc_info=True)

    def get_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding từ OpenAI với retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Lỗi lấy embedding sau {max_retries} lần thử: {str(e)}")
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} cho embedding")
                time.sleep(1)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Tìm context liên quan từ FAISS."""
        try:
            query_embedding = self.get_embedding(query)
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k, len(self.processed_texts))
            )
            threshold = 0.7
            valid_indices = [i for i, d in zip(indices[0], distances[0]) if d < threshold]

            if not valid_indices:
                return "Không tìm thấy context phù hợp."

            contexts = [self.processed_texts[i] for i in valid_indices]
            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            raise

    def get_answer(self, query: str) -> str:
        """Trả lời câu hỏi với cache từ tệp JSON."""
        try:
            if not query.strip():
                return "Vui lòng nhập câu hỏi!"

            # Kiểm tra cache
            if query in self.cache:
                self.logger.info(f"Đã tìm thấy câu trả lời trong cache cho query: {query}")
                return self.cache[query]

            # Lấy context và tạo prompt
            context = self.get_relevant_context(query)
            prompt = f"""Bạn là trợ lý trả lời câu hỏi về điều tra biến động dân số ngày 1/04/2025.
            Hãy trả lời dựa trên context được cung cấp một cách chi tiết và chính xác.
            Nếu không tìm thấy thông tin trong context, hãy nói rõ điều đó.

            Context:
            {context}

            Câu hỏi: {query}

            Trả lời chi tiết dựa trên thông tin trong context:"""

            # Gọi API OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            answer = response.choices[0].message.content

            # Lưu vào cache
            self._save_cache(query, answer)
            return answer

        except Exception as e:
            self.logger.error("Lỗi trong get_answer", exc_info=True)
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return "Hệ thống đang bận, vui lòng thử lại sau ít phút."
            return f"Lỗi xử lý câu hỏi: {error_msg}"

def create_gradio_interface():
    """Tạo giao diện Gradio chỉ hiển thị hỏi đáp."""
    try:
        rag = RAGPipeline()

        def process_question(question: str) -> str:
            try:
                return rag.get_answer(question)
            except Exception as e:
                logging.error("Lỗi trong process_question", exc_info=True)
                return f"Lỗi xử lý câu hỏi: {str(e)}"

        interface = gr.Interface(
            fn=process_question,
            inputs=gr.Textbox(
                lines=2,
                placeholder="Nhập câu hỏi của bạn...",
                label="Câu hỏi"
            ),
            outputs=gr.Textbox(
                lines=10,
                label="Câu trả lời"
            ),
            title="Hệ thống Hỏi Đáp",
            description="Nhập câu hỏi của bạn về nghiệp vụ điều tra thống kê biến động dân số",
            theme="soft",
            examples=[
                ["Quy trình điều tra thống kê gồm những bước nào?"],
                ["Làm thế nào để đảm bảo chất lượng điều tra?"],
            ]
        )
        return interface

    except Exception as e:
        logging.error("Lỗi khởi tạo interface", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv("PORT", 7860)),
            share=False
        )
    except Exception as e:
        logging.error("Lỗi khởi động ứng dụng", exc_info=True)
        print(f"Lỗi khởi động: {str(e)}")