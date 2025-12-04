"""
Тестовый скрипт для генерации эмбеддингов.
"""

import json
import sys
import os
from pathlib import Path

# КРИТИЧНО: Отключаем Xet Storage ДО импорта любых модулей, использующих huggingface_hub
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_XET'] = '0'

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import (
    create_embedding_generator,
    add_embeddings_to_records
)


def test_embeddings_generation():
    """Тестирование генерации эмбеддингов."""
    
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ")
    print("=" * 60)
    
    # Путь к файлу с данными
    input_file = Path(__file__).parent.parent / 'data/output/chunked_data.jsonl'
    
    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        print("Сначала запустите: python tests/test_chunking.py")
        return
    
    # Загружаем данные
    print(f"\n[INFO] Загрузка данных из {input_file.name}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Загружено записей: {len(records)}")
    
    # Показываем первые записи
    print("\n" + "-" * 60)
    print("АНАЛИЗ ДАННЫХ:")
    print("-" * 60)
    
    print(f"Всего записей: {len(records)}")
    
    # Определяем провайдера
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'sentence_transformers').lower()
    
    print("\n" + "-" * 60)
    print("НАСТРОЙКА ГЕНЕРАТОРА ЭМБЕДДИНГОВ:")
    print("-" * 60)
    
    try:
        if embedding_provider == 'openai':
            print("Провайдер: OpenAI")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("[WARNING] OPENAI_API_KEY не найден, переключаемся на Sentence Transformers")
                embedding_provider = 'sentence_transformers'
            else:
                embedding_generator = create_embedding_generator(
                    provider="openai",
                    model="text-embedding-3-small",
                    api_key=api_key
                )
                print(f"Модель: text-embedding-3-small")
                print(f"Размерность: {embedding_generator.get_dimension()}")
        
        if embedding_provider == 'sentence_transformers':
            print("Провайдер: Sentence Transformers (локально, бесплатно)")
            # Используем модель без Xet Storage по умолчанию
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print(f"Модель: {model_name}")
            print("[INFO] Загрузка модели... (это может занять время при первом запуске)")
            
            # Параметры для обработки таймаутов (можно настроить через переменные окружения)
            download_timeout = int(os.getenv('HF_HUB_DOWNLOAD_TIMEOUT', '600'))  # 10 минут по умолчанию
            max_retries = int(os.getenv('HF_DOWNLOAD_MAX_RETRIES', '3'))
            retry_delay = float(os.getenv('HF_DOWNLOAD_RETRY_DELAY', '5.0'))
            
            embedding_generator = create_embedding_generator(
                provider="sentence_transformers",
                model_name=model_name,
                device="cpu",
                download_timeout=download_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            print(f"Размерность эмбеддингов: {embedding_generator.get_dimension()}")
    
    except ImportError as e:
        print(f"[ERROR] Не установлена необходимая библиотека: {e}")
        print("\nУстановите зависимости:")
        if embedding_provider == 'sentence_transformers':
            print("  pip install sentence-transformers")
        else:
            print("  pip install openai")
        return
    except Exception as e:
        print(f"[ERROR] Ошибка при инициализации генератора: {e}")
        return
    
    # Ограничиваем количество записей для теста (опционально)
    test_limit = os.getenv('EMBEDDING_TEST_LIMIT')
    if test_limit:
        limit = int(test_limit)
        print(f"\n[INFO] Ограничение теста: обрабатываем только первые {limit} записей")
        records = records[:limit]
    
    # Генерируем эмбеддинги
    print("\n" + "-" * 60)
    print("ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ...")
    print("-" * 60)
    
    print(f"Обрабатываем {len(records)} записей...")
    print("[INFO] Это может занять некоторое время...")
    
    try:
        records_with_embeddings = add_embeddings_to_records(
            records,
            embedding_generator,
            text_field='text',
            embedding_field='embedding'
        )
        
        print(f"[OK] Эмбеддинги сгенерированы для {len(records_with_embeddings)} записей")
        
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации эмбеддингов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Проверяем результаты
    print("\n" + "-" * 60)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ:")
    print("-" * 60)
    
    records_with_emb = [r for r in records_with_embeddings if r.get('embedding')]
    records_without_emb = [r for r in records_with_embeddings if not r.get('embedding')]
    
    print(f"Записей с эмбеддингами: {len(records_with_emb)}")
    print(f"Записей без эмбеддингов: {len(records_without_emb)}")
    
    if records_with_emb:
        first_emb = records_with_emb[0]['embedding']
        print(f"Размерность эмбеддинга: {len(first_emb)}")
        print(f"Пример первых 5 значений: {first_emb[:5]}")
    
    # Показываем примеры
    print("\n" + "-" * 60)
    print("ПРИМЕРЫ ЗАПИСЕЙ С ЭМБЕДДИНГАМИ:")
    print("-" * 60)
    
    for i, record in enumerate(records_with_emb[:3], 1):
        print(f"\n{i}. ID: {record['id']}")
        print(f"   Вопрос: {record['question']}")
        emb = record['embedding']
        print(f"   Размерность эмбеддинга: {len(emb)}")
        print(f"   Первые значения: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ...]")
    
    # Сохраняем результаты
    output_file = Path(__file__).parent.parent / 'data/output/embeddings_data.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем без эмбеддингов (они большие) в JSONL, но создаем отдельный файл со статистикой
    print("\n" + "-" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    print("-" * 60)
    
    # Сохраняем записи БЕЗ эмбеддингов в JSONL (для экономии места)
    records_for_save = []
    for record in records_with_embeddings:
        record_copy = record.copy()
        embedding = record_copy.pop('embedding', None)
        if embedding:
            record_copy['has_embedding'] = True
            record_copy['embedding_dimension'] = len(embedding)
            # Сохраняем только первые 3 значения для примера
            record_copy['embedding_sample'] = embedding[:3]
        records_for_save.append(record_copy)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records_for_save:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[OK] Данные (без полных эмбеддингов) сохранены в: {output_file}")
    
    # Сохраняем полные данные с эмбеддингами в отдельный файл (опционально)
    save_full = os.getenv('SAVE_FULL_EMBEDDINGS', 'false').lower() == 'true'
    
    if save_full:
        full_output_file = Path(__file__).parent.parent / 'data/output/embeddings_full.json'
        print(f"\n[INFO] Сохранение полных эмбеддингов в {full_output_file.name}...")
        with open(full_output_file, 'w', encoding='utf-8') as f:
            json.dump(records_with_embeddings, f, ensure_ascii=False, indent=2)
        print(f"[OK] Полные данные сохранены")
    else:
        print(f"\n[INFO] Полные эмбеддинги не сохраняются (установите SAVE_FULL_EMBEDDINGS=true для сохранения)")
    
    # Сохраняем статистику
    stats = {
        'total_records': len(records),
        'records_with_embeddings': len(records_with_emb),
        'records_without_embeddings': len(records_without_emb),
        'embedding_dimension': len(records_with_emb[0]['embedding']) if records_with_emb else 0,
        'provider': embedding_provider,
        'model': model_name if embedding_provider == 'sentence_transformers' else 'text-embedding-3-small'
    }
    
    stats_file = output_file.parent / 'embeddings_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Статистика сохранена в: {stats_file.name}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Генерация эмбеддингов завершена успешно!")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("-" * 60)
    print(f"Всего обработано записей: {stats['total_records']}")
    print(f"Эмбеддинги сгенерированы: {stats['records_with_embeddings']}")
    print(f"Размерность эмбеддингов: {stats['embedding_dimension']}")
    print(f"Провайдер: {stats['provider']}")
    print(f"Модель: {stats['model']}")


def main():
    """Главная функция."""
    try:
        test_embeddings_generation()
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
    except Exception as e:
        print(f"\n[ERROR] Ошибка при выполнении теста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


