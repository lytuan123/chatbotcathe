import os
import logging
from typing import List, Dict
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from openai import OpenAI

@dataclass
class ProcessingConfig:
    chunk_size: int = 2000  # Chunk size tối ưu cho chương dài
    overlap: int = 500      # Overlap cho chương dài
    batch_size: int = 16
    max_workers: int = 4
    min_chunk_length: int = 50

class TextProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_logging()
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def setup_logging(self):
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler('rag_pipeline.log', maxBytes=1024*1024, backupCount=5, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def load_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Tải PDF và tách thành các chương"""
        chapters = split_into_chapters(pdf_path)
        self.logger.info(f"Successfully loaded {len(chapters)} chapters from {pdf_path}")
        return self._chunk_documents(chapters)

    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Chia nhỏ văn bản thành các chunk dưới 8192 ký tự"""
        chunks = []
        max_token_length = 8192  # Giới hạn của text-embedding-3-large

        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"].copy()
            metadata["chapter"] = doc["title"]

            if len(text) <= max_token_length:  # Giữ nguyên nếu dưới 8192 ký tự
                chunks.append({"content": text, "metadata": metadata})
            else:
                # Chia nhỏ văn bản thành các đoạn dưới 8192 ký tự
                current_chunk = ""
                current_length = 0

                # Tách theo từ để giữ ngữ nghĩa
                words = text.split()
                for word in words:
                    word_length = len(word) + 1  # +1 cho khoảng trắng
                    if current_length + word_length <= max_token_length - 100:  # Dự phòng 100 ký tự
                        current_chunk += (word + " ")
                        current_length += word_length
                    else:
                        if current_length >= self.config.min_chunk_length:
                            chunks.append({"content": current_chunk.strip(), "metadata": metadata})
                        # Bắt đầu chunk mới, bao gồm overlap
                        current_chunk = " ".join(words[max(0, len(words) - self.config.overlap):len(words)][:self.config.overlap]) + " " + word + " "
                        current_length = len(current_chunk)

                # Thêm chunk cuối nếu đủ dài
                if current_chunk and len(current_chunk) >= self.config.min_chunk_length:
                    chunks.append({"content": current_chunk.strip(), "metadata": metadata})

        return chunks

    def preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\w\s.,;?!-]', ' ', text)
        text = ' '.join(text.split())
        text = re.sub(r'(\d+)\s+(%|đồng|kg|m2|ha)', r'\1\2', text)
        return text.strip()

    def get_embedding(self, text: str) -> np.ndarray:
        instruction = "Represent this text for retrieval of Vietnamese agricultural and statistical survey content: "
        augmented_text = instruction + text
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=augmented_text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def create_embeddings(self, documents: List[Dict[str, str]]) -> tuple:
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            processed_docs = []
            for doc in tqdm(documents, desc="Preprocessing texts"):
                processed_text = executor.submit(self.preprocess_text, doc["content"]).result()
                processed_docs.append({"content": processed_text, "metadata": doc["metadata"]})

        embeddings = []
        for i in tqdm(range(0, len(processed_docs), self.config.batch_size), desc="Creating embeddings"):
            batch = processed_docs[i:i + self.config.batch_size]
            batch_embeddings = [self.get_embedding(doc["content"]) for doc in batch]
            embeddings.extend(batch_embeddings)

        embeddings_array = np.vstack(embeddings)
        dimension = embeddings_array.shape[1]
        nlist = min(len(processed_docs) // 10, 100)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings_array)
        index.train(embeddings_array)
        index.add(embeddings_array)
        self.logger.info(f"Created embeddings for {len(processed_docs)} chunks")
        return index, processed_docs

    def save_artifacts(self, index: faiss.Index, documents: List[Dict[str, str]], output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_dir / 'faiss_index.bin'))
        with open(output_dir / 'processed_texts.pkl', 'wb') as f:
            pickle.dump(documents, f)
        with open(output_dir / 'config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        self.logger.info(f"Saved artifacts to {output_dir}")

def main():
    config = ProcessingConfig()
    processor = TextProcessor(config)
    pdf_path = "/BDDS.pdf"
    documents = processor.load_pdf(pdf_path)
    index, processed_docs = processor.create_embeddings(documents)
    processor.save_artifacts(index, processed_docs, "output")

if __name__ == "__main__":
    main()