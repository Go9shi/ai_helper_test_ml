"""
Демонстрация разбиения на чанки с более агрессивными настройками
для показа работы чанкера на коротких текстах.
"""

import json
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.text_chunker import TextChunker


def demo_chunking():
    """Демонстрация разбиения на чанки с меньшим порогом."""
    
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ РАЗБИЕНИЯ НА ЧАНКИ")
    print("=" * 60)
    
    # Путь к файлу с очищенными данными
    input_file = Path(__file__).parent.parent / 'data/output/cleaned_data.jsonl'
    
    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        return
    
    # Загружаем данные
    print(f"\n[INFO] Загрузка данных из {input_file.name}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"[OK] Загружено записей: {len(records)}")
    
    # Демонстрация: создаем длинный текст для разбиения
    print("\n" + "-" * 60)
    print("СОЗДАНИЕ ДЛИННОГО ТЕКСТА ДЛЯ ДЕМОНСТРАЦИИ:")
    print("-" * 60)
    
    # Берем первую запись и расширяем её ответ для демонстрации
    demo_record = records[0].copy()
    
    # Создаем длинный текст, объединив несколько записей
    long_answer = """
    Как проверить баланс по карте? Вы можете проверить баланс несколькими способами.
    
    Во-первых, через мобильное приложение банка. Откройте приложение на вашем смартфоне,
    войдите в свой аккаунт используя PIN-код или биометрию, и на главном экране вы
    сразу увидите текущий баланс вашей карты. В приложении также доступна детальная
    информация о всех операциях, вы можете фильтровать их по дате, категориям и суммам.
    
    Во-вторых, вы можете проверить баланс через банкомат. Подойдите к любому банкомату
    вашего банка, вставьте карту в картридер, введите PIN-код при запросе. На экране
    появится меню с различными опциями - выберите "Проверить баланс" или "Запрос баланса".
    Банкомат покажет вам текущий баланс на экране, и вы сможете также распечатать чек
    с этой информацией для ваших записей.
    
    В-третьих, доступен онлайн-банк через веб-сайт. Откройте браузер на компьютере или
    планшете, перейдите на официальный сайт вашего банка, войдите в систему используя
    логин и пароль. После входа откройте раздел "Счета и карты" или "Мои финансы", и
    там будет отображен баланс всех ваших карт и счетов в реальном времени.
    
    Также вы всегда можете проверить баланс, позвонив в службу поддержки банка. Наберите
    бесплатный номер горячей линии, следуйте голосовым инструкциям или нажмите соответствующую
    цифру для автоматического запроса баланса. Для этого вам потребуется ввести номер карты
    или идентификационный номер клиента, а также PIN-код или кодовое слово для подтверждения
    личности. Оператор также может помочь вам получить информацию о балансе и последних
    операциях.
    
    Кроме того, многие банки поддерживают SMS-запрос баланса. Отправьте SMS на специальный
    номер банка с определенным текстом команды (например, "БАЛАНС" или "BALANCE"), и вы
    получите ответное SMS с текущим балансом карты. Этот способ удобен когда нет доступа
    к интернету или банкомату.
    """
    
    demo_record['answer'] = long_answer.strip()
    demo_record['text'] = f"Вопрос: {demo_record['question']}\nОтвет: {long_answer.strip()}"
    
    print(f"Создан длинный текст: {len(demo_record['text'])} символов")
    
    # Настройка чанкера с разными параметрами
    configs = [
        {"chunk_size": 500, "chunk_overlap": 50, "name": "Стандартные (500/50)"},
        {"chunk_size": 300, "chunk_overlap": 50, "name": "Средние (300/50)"},
        {"chunk_size": 200, "chunk_overlap": 30, "name": "Маленькие (200/30)"},
    ]
    
    all_results = []
    
    for config in configs:
        print("\n" + "-" * 60)
        print(f"КОНФИГУРАЦИЯ: {config['name']}")
        print("-" * 60)
        
        chunker = TextChunker(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        chunks = chunker.chunk_record(demo_record, enable_chunking=True)
        
        print(f"Создано чанков: {len(chunks)}")
        print(f"Размер исходного текста: {len(demo_record['text'])} символов")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_length = len(chunk['text'])
            print(f"\nЧанк {i} (ID: {chunk['id']}):")
            print(f"  Размер: {chunk_length} символов")
            print(f"  Текст: {chunk['text'][:80].replace(chr(8594), '->')}...")
        
        all_results.append({
            'config': config['name'],
            'chunks': len(chunks),
            'chunks_data': chunks
        })
    
    # Сохраняем результаты демонстрации
    output_file = Path(__file__).parent.parent / 'data/output/chunking_demo.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    demo_output = {
        'original_text_length': len(demo_record['text']),
        'original_question': demo_record['question'],
        'results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "-" * 60)
    print("СОХРАНЕНИЕ ДЕМОНСТРАЦИИ...")
    print("-" * 60)
    print(f"[OK] Результаты сохранены в: {output_file}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Демонстрация завершена!")
    print("=" * 60)
    
    # Итоговая статистика
    print("\n" + "-" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("-" * 60)
    print(f"Исходная длина текста: {len(demo_record['text'])} символов")
    print(f"\nКоличество чанков для разных настроек:")
    for result in all_results:
        print(f"  - {result['config']}: {result['chunks']} чанков")


def main():
    """Главная функция."""
    try:
        demo_chunking()
    except Exception as e:
        print(f"\n[ERROR] Ошибка при выполнении демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



