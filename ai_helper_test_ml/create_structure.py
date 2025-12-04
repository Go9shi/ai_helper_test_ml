"""
Скрипт для создания структуры папок проекта RAG системы.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Создает структуру папок для проекта."""
    
    # Определяем структуру папок
    structure = {
        'data': {
            'raw': 'Исходные данные (Markdown файлы)',
            'processed': 'Обработанные данные (JSON, JSONL)',
            'output': 'Выходные файлы после обработки'
        },
        'scripts': {
            '': 'Скрипты для обработки данных'
        },
        'docs': {
            '': 'Документация проекта'
        },
        'tests': {
            '': 'Тесты для модулей'
        },
        'examples': {
            '': 'Примеры использования'
        },
        'config': {
            '': 'Конфигурационные файлы'
        },
        'logs': {
            '': 'Логи работы системы'
        }
    }
    
    print("=" * 60)
    print("Создание структуры папок проекта")
    print("=" * 60)
    
    base_path = Path('.')
    
    for main_folder, subfolders in structure.items():
        main_path = base_path / main_folder
        
        # Создаем главную папку
        main_path.mkdir(exist_ok=True)
        print(f"\n[OK] Создана папка: {main_folder}/")
        
        # Создаем README в главной папке с описанием
        readme_path = main_path / 'README.md'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {main_folder.capitalize()}\n\n")
                if isinstance(subfolders, dict) and subfolders.get(''):
                    f.write(f"{subfolders['']}\n")
        
        # Создаем подпапки
        if isinstance(subfolders, dict):
            for subfolder, description in subfolders.items():
                if subfolder:  # Пропускаем пустую строку
                    sub_path = main_path / subfolder
                    sub_path.mkdir(exist_ok=True)
                    print(f"  [OK] Создана подпапка: {main_folder}/{subfolder}/")
                    
                    # Создаем README в подпапке
                    sub_readme_path = sub_path / 'README.md'
                    if not sub_readme_path.exists():
                        with open(sub_readme_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {subfolder.capitalize()}\n\n{description}\n")
    
    # Создаем .gitkeep файлы для пустых папок
    empty_folders = [
        'data/raw',
        'data/processed',
        'data/output',
        'logs'
    ]
    
    for folder in empty_folders:
        gitkeep_path = base_path / folder / '.gitkeep'
        gitkeep_path.parent.mkdir(parents=True, exist_ok=True)
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"  [OK] Создан .gitkeep в {folder}/")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Структура папок создана!")
    print("=" * 60)
    
    # Выводим структуру
    print("\nСтруктура проекта:")
    print_tree_structure()


def print_tree_structure():
    """Выводит структуру проекта в виде дерева."""
    structure_text = """
Project structure:
.
+-- data/                      Data files
|   +-- raw/                   Source data (Markdown)
|   +-- processed/             Processed data (JSON, JSONL)
|   +-- output/                Output files
|
+-- rag_pipeline/              RAG system modules
|   +-- __init__.py
|   +-- data_cleaner.py        Data cleaning
|   +-- text_chunker.py        Text chunking
|   +-- embeddings.py          Embeddings generation
|   +-- vector_db.py           Vector database
|   +-- rag_system.py          RAG system
|
+-- scripts/                   Processing scripts
|   +-- markdown_to_rag.py     Markdown conversion
|   +-- rag_pipeline_full.py   Full pipeline
|
+-- examples/                  Usage examples
|   +-- example_rag_usage.py   RAG examples
|
+-- tests/                     Tests
|   +-- test_pipeline.py       Pipeline tests
|
+-- docs/                      Documentation
|   +-- QUICKSTART.md          Quick start
|   +-- README_RAG.md          Basic docs
|   +-- README_RAG_FULL.md     Full docs
|   +-- PROJECT_OVERVIEW.md    Project overview
|
+-- config/                    Configuration
|   +-- config.example.json    Example config
|
+-- logs/                      Logs
|
+-- requirements.txt           Main dependencies
+-- requirements_rag.txt       RAG dependencies
-- README.md                   Main README
"""
    try:
        print(structure_text)
    except:
        # Если не получается вывести, просто показываем список
        print("\nProject structure created successfully!")
        print("See PROJECT_STRUCTURE.md for details")


if __name__ == "__main__":
    create_directory_structure()

