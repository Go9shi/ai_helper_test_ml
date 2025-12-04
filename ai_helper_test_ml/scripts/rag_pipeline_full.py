"""
Полный пайплайн для подготовки данных и загрузки в векторную БД RAG.
Включает: очистку данных, разбиение на чанки, генерацию эмбеддингов и загрузку в БД.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Добавляем корневую директорию в путь для импорта
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

# КРИТИЧНО: Отключаем Xet Storage ДО импорта любых модулей
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_XET'] = '0'
os.environ['SENTENCE_TRANSFORMERS_DISABLE_ONNX'] = '1'
os.environ['ST_DISABLE_ONNX'] = '1'

from rag_pipeline.data_cleaner import DataCleaner
from rag_pipeline.text_chunker import TextChunker
from rag_pipeline.embeddings import (
    create_embedding_generator,
    add_embeddings_to_records
)
from rag_pipeline.vector_db import create_vector_db


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Загружает данные из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Загружает данные из JSONL файла."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_json_file(data: List[Dict[str, Any]], file_path: str):
    """Сохраняет данные в JSON файл."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_full_pipeline(
    input_file: str,
    output_file: str = "rag_processed.json",
    enable_cleaning: bool = True,
    enable_chunking: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_provider: str = "sentence_transformers",
    embedding_model: Optional[str] = None,
    vector_db_provider: Optional[str] = None,
    vector_db_config: Optional[Dict[str, Any]] = None,
    save_embeddings: bool = True,
    upload_to_db: bool = False
):
    """
    Запускает полный пайплайн обработки данных для RAG.
    
    Args:
        input_file: Путь к входному JSON/JSONL файлу
        output_file: Путь к выходному файлу
        enable_cleaning: Включить очистку данных
        enable_chunking: Включить разбиение на чанки
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие между чанками
        embedding_provider: Провайдер эмбеддингов ('openai' или 'sentence_transformers')
        embedding_model: Модель для эмбеддингов
        vector_db_provider: Провайдер векторной БД ('pinecone' или 'qdrant')
        vector_db_config: Конфигурация векторной БД
        save_embeddings: Сохранять ли эмбеддинги в файл
        upload_to_db: Загружать ли в векторную БД
    """
    print("=" * 60)
    print("RAG PIPELINE - Полная обработка данных")
    print("=" * 60)
    
    # 1. Загрузка данных
    print(f"\n[STEP 1] Загрузка данных из {input_file}...")
    if input_file.endswith('.jsonl'):
        records = load_jsonl_file(input_file)
    else:
        records = load_json_file(input_file)
    
    print(f"[OK] Загружено {len(records)} записей")
    
    # 2. Очистка данных
    if enable_cleaning:
        print(f"\n[STEP 2] Очистка данных...")
        cleaner = DataCleaner()
        
        # Получаем статистику до очистки
        stats_before = cleaner.get_statistics(records)
        print(f"  Записей до очистки: {stats_before['total']}")
        
        # Очищаем данные
        records = cleaner.clean_records(records, remove_duplicates=True)
        
        # Статистика после очистки
        stats_after = cleaner.get_statistics(records)
        print(f"[OK] Записей после очистки: {stats_after['total']}")
        print(f"  Категорий: {len(stats_after['categories'])}")
        print(f"  Разделов: {len(stats_after['sections'])}")
    else:
        print(f"\n[STEP 2] Очистка данных пропущена")
    
    # 3. Разбиение на чанки
    if enable_chunking:
        print(f"\n[STEP 3] Разбиение на чанки...")
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        records = chunker.chunk_records(records, enable_chunking=True)
        print(f"[OK] Обработано записей: {len(records)}")
    else:
        print(f"\n[STEP 3] Разбиение на чанки пропущено")
    
    # 4. Генерация эмбеддингов
    print(f"\n[STEP 4] Генерация эмбеддингов ({embedding_provider})...")
    
    embedding_kwargs = {}
    if embedding_provider == "openai":
        if embedding_model:
            embedding_kwargs['model'] = embedding_model
    elif embedding_provider == "sentence_transformers":
        if embedding_model:
            embedding_kwargs['model_name'] = embedding_model
    
    embedding_generator = create_embedding_generator(embedding_provider, **embedding_kwargs)
    records = add_embeddings_to_records(records, embedding_generator)
    
    # Удаляем эмбеддинги из записи перед сохранением (если не нужно)
    if not save_embeddings:
        for record in records:
            if 'embedding' in record:
                del record['embedding']
    
    print(f"[OK] Эмбеддинги сгенерированы")
    
    # 5. Сохранение обработанных данных
    print(f"\n[STEP 5] Сохранение данных в {output_file}...")
    save_json_file(records, output_file)
    print(f"[OK] Данные сохранены")
    
    # 6. Загрузка в векторную БД
    if upload_to_db and vector_db_provider and vector_db_config:
        print(f"\n[STEP 6] Загрузка в векторную БД ({vector_db_provider})...")
        
        vector_db = create_vector_db(vector_db_provider, **vector_db_config)
        success = vector_db.upsert(records)
        
        if success:
            print(f"[OK] Данные загружены в векторную БД")
        else:
            print(f"[ERROR] Ошибка при загрузке в векторную БД")
    else:
        print(f"\n[STEP 6] Загрузка в векторную БД пропущена")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Пайплайн завершен успешно!")
    print("=" * 60)


def main():
    """Главная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Полный пайплайн обработки данных для RAG системы'
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Входной JSON/JSONL файл')
    parser.add_argument('--output', '-o', type=str, default='data/output/rag_processed.json',
                       help='Выходной JSON файл (по умолчанию: data/output/rag_processed.json)')
    
    parser.add_argument('--no-cleaning', action='store_true',
                       help='Отключить очистку данных')
    parser.add_argument('--no-chunking', action='store_true',
                       help='Отключить разбиение на чанки')
    
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Размер чанка (по умолчанию: 500)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Перекрытие между чанками (по умолчанию: 50)')
    
    parser.add_argument('--embedding-provider', type=str, 
                       choices=['openai', 'sentence_transformers'],
                       default='sentence_transformers',
                       help='Провайдер эмбеддингов')
    parser.add_argument('--embedding-model', type=str,
                       help='Модель для эмбеддингов (опционально)')
    
    parser.add_argument('--no-save-embeddings', action='store_true',
                       help='Не сохранять эмбеддинги в выходной файл')
    
    parser.add_argument('--upload-db', action='store_true',
                       help='Загрузить данные в векторную БД')
    parser.add_argument('--db-provider', type=str,
                       choices=['pinecone', 'qdrant'],
                       help='Провайдер векторной БД')
    
    # Параметры для Pinecone
    parser.add_argument('--pinecone-api-key', type=str,
                       help='API ключ Pinecone')
    parser.add_argument('--pinecone-index', type=str,
                       help='Название индекса Pinecone')
    
    # Параметры для Qdrant
    parser.add_argument('--qdrant-url', type=str,
                       help='URL Qdrant сервера')
    parser.add_argument('--qdrant-collection', type=str,
                       help='Название коллекции Qdrant')
    parser.add_argument('--qdrant-api-key', type=str,
                       help='API ключ Qdrant (опционально)')
    
    args = parser.parse_args()
    
    # Формируем конфигурацию векторной БД
    vector_db_config = None
    vector_db_provider = None
    
    if args.upload_db:
        if args.db_provider == 'pinecone':
            if not args.pinecone_api_key or not args.pinecone_index:
                print("[ERROR] Для Pinecone требуется --pinecone-api-key и --pinecone-index")
                return
            vector_db_provider = 'pinecone'
            vector_db_config = {
                'api_key': args.pinecone_api_key,
                'index_name': args.pinecone_index
            }
        elif args.db_provider == 'qdrant':
            if not args.qdrant_url or not args.qdrant_collection:
                print("[ERROR] Для Qdrant требуется --qdrant-url и --qdrant-collection")
                return
            vector_db_provider = 'qdrant'
            vector_db_config = {
                'url': args.qdrant_url,
                'collection_name': args.qdrant_collection,
                'api_key': args.qdrant_api_key
            }
        else:
            print("[ERROR] Укажите --db-provider (pinecone или qdrant)")
            return
    
    # Запускаем пайплайн
    run_full_pipeline(
        input_file=args.input,
        output_file=args.output,
        enable_cleaning=not args.no_cleaning,
        enable_chunking=not args.no_chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        vector_db_provider=vector_db_provider,
        vector_db_config=vector_db_config,
        save_embeddings=not args.no_save_embeddings,
        upload_to_db=args.upload_db
    )


if __name__ == "__main__":
    main()

