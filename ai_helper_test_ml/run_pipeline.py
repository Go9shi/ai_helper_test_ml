"""
Скрипт-обертка для запуска полного RAG пайплайна.
"""

import os
import sys
from pathlib import Path

# Устанавливаем переменные окружения ДО импорта любых модулей
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_XET'] = '0'
os.environ['SENTENCE_TRANSFORMERS_DISABLE_ONNX'] = '1'
os.environ['ST_DISABLE_ONNX'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1200'

# Добавляем текущую директорию в путь
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Импортируем и запускаем пайплайн
from scripts.rag_pipeline_full import run_full_pipeline

if __name__ == "__main__":
    # Параметры по умолчанию
    input_file = "data/processed/rag_data.jsonl"
    output_file = "data/output/rag_processed.json"
    
    print("=" * 60)
    print("ЗАПУСК ПОЛНОГО RAG ПАЙПЛАЙНА")
    print("=" * 60)
    print(f"Входной файл: {input_file}")
    print(f"Выходной файл: {output_file}")
    print()
    
    # Запускаем пайплайн
    run_full_pipeline(
        input_file=input_file,
        output_file=output_file,
        enable_cleaning=True,
        enable_chunking=True,
        chunk_size=500,
        chunk_overlap=50,
        embedding_provider="sentence_transformers",
        embedding_model=None,
        vector_db_provider=None,
        vector_db_config=None,
        save_embeddings=True,
        upload_to_db=False
    )

