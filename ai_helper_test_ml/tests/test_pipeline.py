"""
Тестовый скрипт для проверки работы пайплайна обработки данных.
Работает без подключения к векторным БД - только локальная обработка.
"""

import json
from rag_pipeline.data_cleaner import DataCleaner
from rag_pipeline.text_chunker import TextChunker


def test_data_cleaning():
    """Тестирование очистки данных."""
    print("\n" + "=" * 60)
    print("ТЕСТ: Очистка данных")
    print("=" * 60)
    
    # Тестовые данные
    test_records = [
        {
            "id": "1",
            "question": "Как проверить баланс?",
            "category": "Общие операции",
            "answer": "Проверьте в приложении.",
            "text": "Вопрос: Как проверить баланс?\nОтвет: Проверьте в приложении.",
            "section": "Общие операции"
        },
        {
            "id": "2",
            "question": "Как проверить баланс?  ",  # С лишними пробелами
            "category": "Общие операции",
            "answer": "Проверьте в приложении.  ",  # Дубликат с опечатками
            "text": "Вопрос: Как проверить баланс?  \nОтвет: Проверьте в приложении.  ",
            "section": "Общие операции"
        },
        {
            "id": "3",
            "question": "Как заблокировать карту?",
            "category": "Безопасность",
            "answer": "В приложении или по телефону.",
            "text": "Вопрос: Как заблокировать карту?\nОтвет: В приложении или по телефону.",
            "section": "Безопасность"
        }
    ]
    
    print(f"Записей до очистки: {len(test_records)}")
    
    cleaner = DataCleaner()
    cleaned = cleaner.clean_records(test_records, remove_duplicates=True)
    
    print(f"Записей после очистки: {len(cleaned)}")
    
    # Статистика
    stats = cleaner.get_statistics(cleaned)
    print(f"\nСтатистика:")
    print(f"  Всего записей: {stats['total']}")
    print(f"  Категорий: {len(stats['categories'])}")
    print(f"  Средняя длина вопроса: {stats['avg_question_length']:.1f} символов")
    
    print("\n[OK] Тест очистки данных пройден")
    return cleaned


def test_chunking():
    """Тестирование разбиения на чанки."""
    print("\n" + "=" * 60)
    print("ТЕСТ: Разбиение на чанки")
    print("=" * 60)
    
    # Длинный текст для разбиения
    long_text = """
    Как проверить баланс по карте? Вы можете проверить баланс несколькими способами.
    Во-первых, в мобильном приложении банка. Откройте приложение, войдите в свой аккаунт
    и на главном экране вы увидите текущий баланс вашей карты. Во-вторых, вы можете
    проверить баланс через банкомат. Вставьте карту, введите PIN-код и выберите
    опцию "Проверить баланс". В-третьих, вы можете использовать онлайн-банк. Войдите
    на сайт банка, откройте раздел со счетами и картами, и там будет отображен баланс.
    Также вы можете проверить баланс, позвонив в службу поддержки банка.
    """
    
    test_record = {
        "id": "test_1",
        "question": "Как проверить баланс?",
        "category": "Общие операции",
        "answer": long_text.strip(),
        "text": f"Вопрос: Как проверить баланс?\nОтвет: {long_text.strip()}",
        "section": "Общие операции"
    }
    
    print(f"Длина исходного текста: {len(test_record['text'])} символов")
    
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_record(test_record, enable_chunking=True)
    
    print(f"Создано чанков: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nЧанк {i} (ID: {chunk['id']}):")
        print(f"  Длина: {len(chunk['text'])} символов")
        print(f"  Текст: {chunk['text'][:100]}...")
    
    print("\n[OK] Тест разбиения на чанки пройден")
    return chunks


def test_full_pipeline():
    """Полный тест пайплайна (без эмбеддингов и БД)."""
    print("\n" + "=" * 60)
    print("ТЕСТ: Полный пайплайн (очистка + чанки)")
    print("=" * 60)
    
    # Загружаем реальные данные если есть
    try:
        with open('rag_data.jsonl', 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"Загружено {len(records)} записей из rag_data.jsonl")
    except FileNotFoundError:
        print("[INFO] Файл rag_data.jsonl не найден, используем тестовые данные")
        records = [
            {
                "id": "1",
                "question": "Как проверить баланс по карте?",
                "category": "Общие операции",
                "answer": "Проверьте в мобильном приложении, банкомате или онлайн-банке.",
                "text": "Вопрос: Как проверить баланс по карте?\nОтвет: Проверьте в мобильном приложении, банкомате или онлайн-банке.",
                "section": "Общие операции"
            }
        ]
    
    # Шаг 1: Очистка
    print("\n[1/2] Очистка данных...")
    cleaner = DataCleaner()
    cleaned = cleaner.clean_records(records[:5], remove_duplicates=True)  # Тестируем на 5 записях
    print(f"     Записей после очистки: {len(cleaned)}")
    
    # Шаг 2: Разбиение на чанки (только для длинных)
    print("\n[2/2] Разбиение на чанки...")
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunked = chunker.chunk_records(cleaned, enable_chunking=True)
    print(f"     Записей после чанкинга: {len(chunked)}")
    
    # Сохраняем результат
    output_file = 'test_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Результат сохранен в {output_file}")
    
    return chunked


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ RAG PIPELINE")
    print("=" * 60)
    
    try:
        # Тест 1: Очистка данных
        test_data_cleaning()
        
        # Тест 2: Разбиение на чанки
        test_chunking()
        
        # Тест 3: Полный пайплайн
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Все тесты пройдены успешно!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

