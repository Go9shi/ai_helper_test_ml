"""
Скрипт для организации существующих файлов в структуру папок.
"""

import shutil
from pathlib import Path


def organize_files():
    """Организует существующие файлы по структуре папок."""
    
    print("=" * 60)
    print("Организация файлов по структуре")
    print("=" * 60)
    
    # Определяем перемещения файлов
    file_movements = {
        # Исходные данные
        'test_asks.md': 'data/raw/',
        
        # Обработанные данные
        'rag_data.json': 'data/processed/',
        'rag_data.jsonl': 'data/processed/',
        'rag_data.csv': 'data/processed/',
        
        # Скрипты
        'markdown_to_rag.py': 'scripts/',
        'rag_pipeline_full.py': 'scripts/',
        
        # Примеры
        'example_rag_usage.py': 'examples/',
        
        # Тесты
        'test_pipeline.py': 'tests/',
        
        # Документация
        'QUICKSTART.md': 'docs/',
        'README_RAG.md': 'docs/',
        'README_RAG_FULL.md': 'docs/',
        'PROJECT_OVERVIEW.md': 'docs/',
        
        # Конфигурация
        'requirements_rag.txt': '.',  # Оставляем в корне
    }
    
    moved_files = []
    skipped_files = []
    
    for file_name, target_dir in file_movements.items():
        source_path = Path(file_name)
        target_path = Path(target_dir) / file_name
        
        if source_path.exists():
            try:
                # Создаем целевую директорию если её нет
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Перемещаем файл
                shutil.move(str(source_path), str(target_path))
                moved_files.append((file_name, target_dir))
                print(f"[OK] Перемещен: {file_name} -> {target_dir}")
            except Exception as e:
                print(f"[ERROR] Ошибка при перемещении {file_name}: {e}")
                skipped_files.append(file_name)
        else:
            skipped_files.append(file_name)
            print(f"[SKIP] Файл не найден: {file_name}")
    
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Перемещено файлов: {len(moved_files)}")
    print(f"[SUMMARY] Пропущено файлов: {len(skipped_files)}")
    print("=" * 60)
    
    if moved_files:
        print("\nПеремещенные файлы:")
        for file_name, target_dir in moved_files:
            print(f"  - {file_name} -> {target_dir}/")
    
    if skipped_files:
        print("\nПропущенные файлы (не найдены или ошибка):")
        for file_name in skipped_files:
            print(f"  - {file_name}")


if __name__ == "__main__":
    organize_files()

