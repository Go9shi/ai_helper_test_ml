"""
Тестовый скрипт для разбиения данных на чанки.
"""

import json
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.text_chunker import TextChunker


def test_chunking_from_dataset():
    """Тестирование разбиения данных на чанки."""
    
    print("\n" + "=" * 60)
    print("РАЗБИЕНИЕ ДАННЫХ НА ЧАНКИ")
    print("=" * 60)
    
    # Путь к файлу с очищенными данными
    input_file = Path(__file__).parent.parent / 'data/output/cleaned_data.jsonl'
    
    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        print("Сначала запустите: python tests/test_data_cleaning.py")
        return
    
    # Загружаем данные
    print(f"\n[INFO] Загрузка данных из {input_file.name}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Загружено записей: {len(records)}")
    
    # Анализ длины текстов
    print("\n" + "-" * 60)
    print("АНАЛИЗ ДЛИНЫ ТЕКСТОВ:")
    print("-" * 60)
    
    text_lengths = [len(record.get('text', '')) for record in records]
    max_length = max(text_lengths) if text_lengths else 0
    min_length = min(text_lengths) if text_lengths else 0
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    print(f"Минимальная длина: {min_length} символов")
    print(f"Максимальная длина: {max_length} символов")
    print(f"Средняя длина: {avg_length:.1f} символов")
    
    # Определяем, сколько текстов длиннее стандартного чанка
    chunk_size = 500
    long_texts = [r for r in records if len(r.get('text', '')) > chunk_size]
    print(f"\nТекстов длиннее {chunk_size} символов: {len(long_texts)}")
    
    # Настройка чанкера
    print("\n" + "-" * 60)
    print("НАСТРОЙКА ЧАНКЕРА:")
    print("-" * 60)
    
    chunk_size = 500
    chunk_overlap = 50
    min_text_length = 500  # Минимальная длина для разбиения
    
    print(f"Размер чанка: {chunk_size} символов")
    print(f"Перекрытие: {chunk_overlap} символов")
    print(f"Минимальная длина текста для разбиения: {min_text_length} символов")
    
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Разбиваем на чанки
    print("\n" + "-" * 60)
    print("РАЗБИЕНИЕ НА ЧАНКИ...")
    print("-" * 60)
    
    chunked_records = []
    total_chunks_created = 0
    records_chunked = 0
    
    for record in records:
        original_text_length = len(record.get('text', ''))
        
        if original_text_length > min_text_length:
            # Разбиваем на чанки
            chunks = chunker.chunk_record(record, enable_chunking=True)
            chunked_records.extend(chunks)
            
            if len(chunks) > 1:
                records_chunked += 1
                total_chunks_created += len(chunks) - 1
                print(f"[CHUNKED] ID {record['id']}: {original_text_length} символов -> {len(chunks)} чанков")
        else:
            # Короткие записи оставляем как есть
            record['is_chunk'] = False
            if 'metadata' not in record:
                record['metadata'] = {}
            record['metadata']['is_chunk'] = False
            chunked_records.append(record)
    
    # Статистика после разбиения
    print("\n" + "-" * 60)
    print("СТАТИСТИКА ПОСЛЕ РАЗБИЕНИЯ:")
    print("-" * 60)
    
    print(f"Исходных записей: {len(records)}")
    print(f"Записей после разбиения: {len(chunked_records)}")
    print(f"Создано дополнительных чанков: {total_chunks_created}")
    print(f"Записей разбито на чанки: {records_chunked}")
    print(f"Записей оставлено без изменений: {len(records) - records_chunked}")
    
    # Анализ размеров чанков
    chunk_sizes = [len(r.get('text', '')) for r in chunked_records]
    print(f"\nАнализ размеров чанков:")
    print(f"  Минимальный размер: {min(chunk_sizes)} символов")
    print(f"  Максимальный размер: {max(chunk_sizes)} символов")
    print(f"  Средний размер: {sum(chunk_sizes) / len(chunk_sizes):.1f} символов")
    
    # Показываем примеры чанков
    print("\n" + "-" * 60)
    print("ПРИМЕРЫ СОЗДАННЫХ ЧАНКОВ:")
    print("-" * 60)
    
    chunked_examples = [r for r in chunked_records if r.get('is_chunk', False)][:5]
    
    if chunked_examples:
        for i, chunk in enumerate(chunked_examples, 1):
            print(f"\n{i}. Чанк ID: {chunk['id']}")
            print(f"   Родитель ID: {chunk.get('parent_id', 'N/A')}")
            print(f"   Индекс чанка: {chunk.get('chunk_index', 'N/A')}")
            print(f"   Размер: {len(chunk['text'])} символов")
            print(f"   Текст: {chunk['text'][:100].replace(chr(8594), '->')}...")
    else:
        print("Нет чанков для отображения (все тексты короткие)")
        # Показываем несколько исходных записей
        for i, record in enumerate(records[:3], 1):
            print(f"\n{i}. Запись ID: {record['id']}")
            print(f"   Размер: {len(record['text'])} символов")
            print(f"   Вопрос: {record['question']}")
    
    # Сохраняем результат
    output_file = Path(__file__).parent.parent / 'data/output/chunked_data.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    print("-" * 60)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in chunked_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[OK] Разбитые данные сохранены в: {output_file}")
    print(f"[OK] Сохранено записей: {len(chunked_records)}")
    
    # Также сохраняем в JSON для удобства просмотра
    output_json = output_file.with_suffix('.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunked_records, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Также сохранено в JSON: {output_json.name}")
    
    # Сохраняем статистику
    stats = {
        'original_records': len(records),
        'chunked_records': len(chunked_records),
        'chunks_created': total_chunks_created,
        'records_chunked': records_chunked,
        'records_unchanged': len(records) - records_chunked,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    }
    
    stats_file = output_file.parent / 'chunking_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Статистика сохранена в: {stats_file.name}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Разбиение на чанки завершено успешно!")
    print("=" * 60)
    
    return chunked_records


def main():
    """Главная функция."""
    try:
        test_chunking_from_dataset()
    except Exception as e:
        print(f"\n[ERROR] Ошибка при выполнении теста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



