"""
Тестовый скрипт для очистки данных из реального датасета.
"""

import json
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.data_cleaner import DataCleaner


def test_cleaning_from_dataset():
    """Тестирование очистки данных из реального датасета."""
    
    print("\n" + "=" * 60)
    print("ТЕСТОВАЯ ОЧИСТКА ДАННЫХ ИЗ ДАТАСЕТА")
    print("=" * 60)
    
    # Путь к файлу с данными
    input_file = Path(__file__).parent.parent / 'data/processed/rag_data.jsonl'
    
    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        print("Сначала запустите: python scripts/markdown_to_rag.py")
        return
    
    # Загружаем данные
    print(f"\n[INFO] Загрузка данных из {input_file.name}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Загружено записей: {len(records)}")
    
    # Показываем статистику до очистки
    print("\n" + "-" * 60)
    print("СТАТИСТИКА ДО ОЧИСТКИ:")
    print("-" * 60)
    
    cleaner = DataCleaner()
    stats_before = cleaner.get_statistics(records)
    
    print(f"Всего записей: {stats_before['total']}")
    print(f"Категорий: {len(stats_before['categories'])}")
    print(f"Разделов: {len(stats_before['sections'])}")
    print(f"Средняя длина вопроса: {stats_before['avg_question_length']:.1f} символов")
    print(f"Средняя длина ответа: {stats_before['avg_answer_length']:.1f} символов")
    
    # Показываем распределение по категориям
    print(f"\nРаспределение по категориям:")
    for cat, count in sorted(stats_before['categories'].items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}")
    
    # Очищаем данные
    print("\n" + "-" * 60)
    print("ОЧИСТКА ДАННЫХ...")
    print("-" * 60)
    
    cleaned_records = cleaner.clean_records(records, remove_duplicates=True)
    
    # Статистика после очистки
    print("\n" + "-" * 60)
    print("СТАТИСТИКА ПОСЛЕ ОЧИСТКИ:")
    print("-" * 60)
    
    stats_after = cleaner.get_statistics(cleaned_records)
    
    print(f"Всего записей: {stats_after['total']}")
    print(f"Категорий: {len(stats_after['categories'])}")
    print(f"Разделов: {len(stats_after['sections'])}")
    print(f"Средняя длина вопроса: {stats_after['avg_question_length']:.1f} символов")
    print(f"Средняя длина ответа: {stats_after['avg_answer_length']:.1f} символов")
    
    # Показываем изменения
    print("\n" + "-" * 60)
    print("ИЗМЕНЕНИЯ:")
    print("-" * 60)
    
    removed = stats_before['total'] - stats_after['total']
    print(f"Удалено дубликатов: {removed}")
    print(f"Удалено записей: {removed} ({removed/stats_before['total']*100:.1f}%)")
    
    # Показываем примеры очищенных записей
    print("\n" + "-" * 60)
    print("ПРИМЕРЫ ОЧИЩЕННЫХ ЗАПИСЕЙ:")
    print("-" * 60)
    
    for i, record in enumerate(cleaned_records[:5], 1):
        print(f"\n{i}. ID: {record['id']}")
        print(f"   Вопрос: {record['question']}")
        print(f"   Категория: {record['category']}")
        answer_preview = record['answer'][:80].replace('→', '->')
        print(f"   Ответ: {answer_preview}...")
    
    # Сохраняем очищенные данные
    output_file = Path(__file__).parent.parent / 'data/output/cleaned_data.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    print("-" * 60)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[OK] Очищенные данные сохранены в: {output_file}")
    print(f"[OK] Сохранено записей: {len(cleaned_records)}")
    
    # Также сохраняем в JSON для удобства просмотра
    output_json = output_file.with_suffix('.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Также сохранено в JSON: {output_json.name}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Тестовая очистка завершена успешно!")
    print("=" * 60)
    
    return cleaned_records


def main():
    """Главная функция."""
    try:
        test_cleaning_from_dataset()
    except Exception as e:
        print(f"\n[ERROR] Ошибка при выполнении теста: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

