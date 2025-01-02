import os
import logging
import json
import pickle
import numpy as np
import openai
import faiss
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Cấu hình logging
log_dir = "logs"
log_file = os.path.join(log_dir, "rag_pipeline.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Tạo thư mục cache và history
cache_file = "cache/cache_cache.json"
history_file = "history/chat_history.json"
os.makedirs("cache", exist_ok=True)
os.makedirs("history", exist_ok=True)

# Đảm bảo file cache và history tồn tại
if not os.path.exists(cache_file):
    with open(cache_file, "w") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

if not os.path.exists(history_file):
    with open(history_file, "w") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# Tải API Key từ .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Khởi tạo Flask app
app = Flask(__name__)

# Giới hạn lịch sử
MAX_HISTORY_LENGTH = 10

# Quản lý model
class ModelManager:
    def __init__(self):
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cache từ file."""
        with open(cache_file, "r") as f:
            return json.load(f)

    def save_cache(self):
        """Lưu cache vào file."""
        with open(cache_file, "w") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def call_openai_model(self, prompt: str) -> str:
        """Gọi API OpenAI."""
        if prompt in self.cache:
            logging.info(f"Cache hit for prompt: {prompt}")
            return self.cache[prompt]

        try:
            logging.info(f"Calling OpenAI API for prompt: {prompt}")
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Sử dụng mô hình GPT-4o-mini như yêu cầu
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
                request_timeout=30
            )
            answer = response.choices[0].message["content"]
            self.cache[prompt] = answer
            self.save_cache()
            logging.info(f"Received answer from OpenAI: {answer}")
            return answer
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return "Error: Unable to fetch response from OpenAI."

# RAG Pipeline
class RAGPipeline:
    def __init__(self):
        try:
            print("Initializing RAG Pipeline...")
            logging.info("Initializing RAG Pipeline...")
            if not os.path.exists("faiss_index.bin") or not os.path.exists("chunked_texts.pkl"):
                raise FileNotFoundError("Missing required files: faiss_index.bin or chunked_texts.pkl")

            print("Loading FAISS index...")
            logging.info("Loading FAISS index...")
            self.index = faiss.read_index("faiss_index.bin")

            print("Loading chunked texts...")
            logging.info("Loading chunked texts...")
            with open("chunked_texts.pkl", "rb") as f:
                self.texts = pickle.load(f)

            print("Initializing Model Manager...")
            logging.info("Initializing Model Manager...")
            self.model_manager = ModelManager()

            self.prompt_template = """Trả lời câu hỏi sau dựa trên ngữ cảnh:
{context}

Câu hỏi: {question}
"""
            print("RAG Pipeline initialized successfully")
            logging.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing RAG Pipeline: {e}")
            print(f"Error initializing RAG Pipeline: {e}")
            raise RuntimeError("Error initializing RAG Pipeline.")

    def get_query_embedding(self, query: str) -> np.ndarray:
        try:
            logging.info(f"Fetching embedding for query: {query}")
            response = openai.Embedding.create(
                input=query,
                model="text-embedding-3-large"  # Sử dụng mô hình embedding yêu cầu
            )
            embedding = np.array(response["data"][0]["embedding"], dtype="float32")
            logging.info(f"Received embedding: {embedding[:5]}...")  # Log phần đầu embedding
            return embedding
        except openai.error.OpenAIError as e:
            logging.error(f"Error fetching embedding: {e}")
            raise RuntimeError("Failed to get embedding.")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        logging.info(f"Getting relevant context for query: {query}")
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        contexts = [self.texts[i] for i in indices[0]]
        context_str = "\n\n".join(contexts)
        logging.info(f"Relevant context: {context_str}")
        return context_str

    def get_answer(self, query: str) -> str:
        logging.info(f"Generating answer for query: {query}")
        context = self.get_relevant_context(query)
        prompt = self.prompt_template.format(context=context, question=query)
        logging.info(f"Prompt: {prompt}")
        return self.model_manager.call_openai_model(prompt)

# Khởi tạo pipeline
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    logging.error(f"Failed to initialize RAG Pipeline: {e}")
    print(f"Failed to initialize RAG Pipeline: {e}")
    exit(1)  # Dừng ứng dụng nếu không khởi tạo được pipeline

# Endpoint Flask
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask app is running. Use /answer with POST to query."})

@app.route('/answer', methods=['POST'])
def answer():
    try:
        logging.info("Endpoint '/answer' hit")
        
        # Log raw request data
        raw_data = request.get_data()
        logging.info(f"Raw request data: {raw_data}")
        
        # Verify JSON parsing
        try:
            data = request.get_json()
            logging.info(f"Parsed JSON data: {data}")
        except Exception as e:
            logging.error(f"JSON parsing error: {e}")
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
            
        # Check for query parameter
        query = data.get('query')
        if not query:
            logging.warning("Missing 'query' parameter in request")
            return jsonify({"error": "Missing 'query' parameter"}), 400
            
        # Verify required files exist
        required_files = ["faiss_index.bin", "chunked_texts.pkl"]
        for file in required_files:
            if not os.path.exists(file):
                error_msg = f"Required file {file} not found"
                logging.error(error_msg)
                return jsonify({"error": error_msg}), 500
                
        # Process query
        logging.info(f"Processing query: {query}")
        answer = rag_pipeline.get_answer(query)
        logging.info(f"Generated answer: {answer}")

        # Update history
        try:
            with open(history_file, "r+") as f:
                history = json.load(f)
                history.append({"user": query, "bot": answer})
                if len(history) > MAX_HISTORY_LENGTH:
                    history = history[-MAX_HISTORY_LENGTH:]
                f.seek(0)
                f.truncate()
                json.dump(history, f, ensure_ascii=False, indent=4)
            logging.info("History updated successfully")
        except Exception as e:
            logging.error(f"Error updating history: {e}")
            # Continue even if history update fails
            
        return jsonify({"query": query, "answer": answer})
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Unexpected error occurred: {str(e)}"}), 500
# Chạy ứng dụng Flask
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Lấy cổng từ biến môi trường hoặc mặc định là 5000
    app.run(host='0.0.0.0', port=port, debug=False)
