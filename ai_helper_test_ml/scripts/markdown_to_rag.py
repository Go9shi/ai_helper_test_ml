"""
Скрипт для преобразования Markdown файлов в структурированный формат для загрузки в векторную БД RAG.
Поддерживает таблицы Markdown и извлекает структурированные данные.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def parse_markdown_tables(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Парсит Markdown файл и извлекает данные из таблиц.
    
    Args:
        markdown_content: Содержимое Markdown файла
        
    Returns:
        Список словарей с извлеченными данными
    """
    chunks = []
    lines = markdown_content.split('\n')
    
    current_section = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Определяем заголовок раздела (##)
        if line.startswith('##'):
            current_section = line.replace('##', '').strip()
            i += 1
            continue
        
        # Пропускаем пустые строки
        if not line:
            i += 1
            continue
        
        # Ищем начало таблицы (строка с | ID |)
        if line.startswith('|') and 'ID' in line:
            # Пропускаем разделитель таблицы (|---|---|)
            i += 1
            if i < len(lines) and '---' in lines[i]:
                i += 1
            
            # Читаем строки таблицы
            while i < len(lines):
                table_line = lines[i].strip()
                
                # Если строка не начинается с |, значит таблица закончилась
                if not table_line.startswith('|'):
                    break
                
                # Пропускаем пустые строки таблицы
                if not table_line or table_line == '|' or '---' in table_line:
                    i += 1
                    continue
                
                # Парсим строку таблицы
                parts = [p.strip() for p in table_line.split('|') if p.strip()]
                
                if len(parts) >= 4:
                    try:
                        record_id = parts[0]
                        question = parts[1]
                        category = parts[2]
                        answer = parts[3]
                        
                        # Создаем структурированную запись
                        chunk = {
                            "id": record_id,
                            "question": question,
                            "category": category,
                            "answer": answer,
                            "section": current_section or "Неизвестный раздел",
                            "text": f"Вопрос: {question}\nОтвет: {answer}",  # Текст для эмбеддинга
                            "metadata": {
                                "section": current_section or "Неизвестный раздел",
                                "category": category,
                                "source": "test_asks.md"
                            }
                        }
                        chunks.append(chunk)
                    except (ValueError, IndexError) as e:
                        print(f"Ошибка при парсинге строки: {table_line}, ошибка: {e}")
                
                i += 1
        else:
            i += 1
    
    return chunks


def save_to_json(chunks: List[Dict[str, Any]], output_file: str):
    """Сохраняет данные в JSON файл."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[OK] Сохранено {len(chunks)} записей в {output_file}")


def save_to_jsonl(chunks: List[Dict[str, Any]], output_file: str):
    """Сохраняет данные в JSONL файл (каждая строка - отдельный JSON объект)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"[OK] Сохранено {len(chunks)} записей в {output_file}")


def save_to_csv(chunks: List[Dict[str, Any]], output_file: str):
    """Сохраняет данные в CSV файл."""
    df = pd.DataFrame(chunks)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[OK] Сохранено {len(chunks)} записей в {output_file}")


def main():
    """Основная функция для преобразования Markdown в RAG формат."""
    parser = argparse.ArgumentParser(
        description='Преобразование Markdown файлов в структурированный формат для RAG'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/test_asks.md',
        help='Входной Markdown файл (по умолчанию: data/raw/test_asks.md)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed/rag_data',
        help='Базовое имя выходных файлов без расширения (по умолчанию: data/processed/rag_data)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl', 'csv', 'all'],
        default='all',
        help='Формат выходных файлов (по умолчанию: all)'
    )
    
    args = parser.parse_args()
    
    input_file = args.input
    base_output = args.output
    output_json = f"{base_output}.json"
    output_jsonl = f"{base_output}.jsonl"
    output_csv = f"{base_output}.csv"
    
    # Проверяем существование входного файла
    if not Path(input_file).exists():
        print(f"[ERROR] Файл {input_file} не найден!")
        return
    
    # Читаем Markdown файл
    print(f"[INFO] Чтение файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Парсим Markdown
    print("[INFO] Парсинг Markdown таблиц...")
    chunks = parse_markdown_tables(markdown_content)
    
    if not chunks:
        print("[ERROR] Не удалось извлечь данные из файла!")
        return
    
    print(f"[OK] Извлечено {len(chunks)} записей")
    
    # Сохраняем в различные форматы
    print("\n[INFO] Сохранение данных...")
    if args.format in ['json', 'all']:
        save_to_json(chunks, output_json)
    if args.format in ['jsonl', 'all']:
        save_to_jsonl(chunks, output_jsonl)
    if args.format in ['csv', 'all']:
        save_to_csv(chunks, output_csv)
    
    # Выводим статистику
    print("\n[STATS] Статистика:")
    categories = {}
    sections = {}
    for chunk in chunks:
        cat = chunk['category']
        sec = chunk['section']
        categories[cat] = categories.get(cat, 0) + 1
        sections[sec] = sections.get(sec, 0) + 1
    
    print(f"  Категорий: {len(categories)}")
    print(f"  Разделов: {len(sections)}")
    print(f"\n  Распределение по категориям:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    - {cat}: {count}")
    
    print(f"\n[SUCCESS] Преобразование завершено!")
    print(f"\n[FILES] Созданные файлы:")
    if args.format in ['json', 'all']:
        print(f"  - {output_json} (JSON формат)")
    if args.format in ['jsonl', 'all']:
        print(f"  - {output_jsonl} (JSONL формат, для пакетной загрузки)")
    if args.format in ['csv', 'all']:
        print(f"  - {output_csv} (CSV формат)")


if __name__ == "__main__":
    main()

