"""
Упрощенный скрипт для быстрой генерации эмбеддингов.
Использует более легкую модель для быстрого тестирования.
"""

import json
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import (
    create_embedding_generator,
    add_embeddings_to_records
)


def test_embeddings_quick():
    """Быстрая генерация эмбеддингов для тестирования."""
    
    print("\n" + "=" * 60)
    print("БЫСТРАЯ ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ")
    print("=" * 60)
    
    # Путь к файлу с данными
    input_file = Path(__file__).parent.parent / 'data/output/chunked_data.jsonl'
    
    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        return
    
    # Загружаем только первые 3 записи для быстрого теста
    print(f"\n[INFO] Загрузка данных из {input_file.name}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip() and i < 3:  # Только первые 3 записи
                records.append(json.loads(line))
    
    print(f"[OK] Загружено записей для теста: {len(records)}")
    
    # Используем более легкую модель
    print("\n" + "-" * 60)
    print("ИНИЦИАЛИЗАЦИЯ ГЕНЕРАТОРА:")
    print("-" * 60)
    
    try:
        # Используем более легкую модель для быстрого тестирования
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'  # Быстрая мультиязычная модель
        
        print(f"Провайдер: Sentence Transformers")
        print(f"Модель: sentence-transformers/{model_name}")
        print("[INFO] Загрузка модели (может занять время при первом запуске)...")
        
        embedding_generator = create_embedding_generator(
            provider="sentence_transformers",
            model_name=f'sentence-transformers/{model_name}',
            device="cpu"
        )
        
        print(f"[OK] Модель загружена")
        print(f"[OK] Размерность эмбеддингов: {embedding_generator.get_dimension()}")
    
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Генерируем эмбеддинги
    print("\n" + "-" * 60)
    print("ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ...")
    print("-" * 60)
    
    try:
        records_with_embeddings = add_embeddings_to_records(
            records,
            embedding_generator,
            text_field='text',
            embedding_field='embedding'
        )
        
        print(f"[OK] Эмбеддинги сгенерированы для {len(records_with_embeddings)} записей")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Показываем результаты
    print("\n" + "-" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("-" * 60)
    
    for i, record in enumerate(records_with_embeddings, 1):
        emb = record.get('embedding', [])
        if emb:
            print(f"\n{i}. ID: {record['id']}")
            print(f"   Вопрос: {record['question']}")
            print(f"   Размерность: {len(emb)}")
            print(f"   Первые 5 значений: {[f'{v:.4f}' for v in emb[:5]]}")
    
    # Сохраняем результаты
    output_file = Path(__file__).parent.parent / 'data/output/embeddings_quick_test.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем с полными эмбеддингами для теста
    output_data = []
    for record in records_with_embeddings:
        record_copy = {
            'id': record['id'],
            'question': record['question'],
            'text': record['text'],
            'embedding_dimension': len(record.get('embedding', [])),
            'embedding': record.get('embedding', [])
        }
        output_data.append(record_copy)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Результаты сохранены в: {output_file}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Быстрый тест завершен!")
    print("=" * 60)
    print("\nДля полной генерации запустите: python tests/test_embeddings.py")


if __name__ == "__main__":
    try:
        test_embeddings_quick()
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
    except Exception as e:
        print(f"\n[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()



