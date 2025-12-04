# Структура проекта RAG системы

## Обзор структуры

```
ai_helper/
│
├── 📁 data/                           # Данные проекта
│   ├── 📁 raw/                        # Исходные данные
│   │   ├── test_asks.md              # Исходный Markdown файл
│   │   └── .gitkeep                  # Для Git
│   │
│   ├── 📁 processed/                  # Обработанные данные
│   │   ├── rag_data.json             # JSON формат
│   │   ├── rag_data.jsonl            # JSONL формат (для загрузки)
│   │   ├── rag_data.csv              # CSV формат
│   │   └── .gitkeep
│   │
│   └── 📁 output/                     # Выходные файлы
│       └── .gitkeep
│
├── 📁 rag_pipeline/                   # Модули RAG системы
│   ├── __init__.py                   # Экспорт модулей
│   ├── data_cleaner.py               # Очистка данных
│   ├── text_chunker.py               # Разбиение на чанки
│   ├── embeddings.py                 # Генерация эмбеддингов
│   ├── vector_db.py                  # Работа с векторными БД
│   └── rag_system.py                 # RAG система (поиск + генерация)
│
├── 📁 scripts/                        # Скрипты обработки
│   ├── markdown_to_rag.py            # Преобразование Markdown → JSON/JSONL
│   └── rag_pipeline_full.py          # Полный пайплайн обработки
│
├── 📁 examples/                       # Примеры использования
│   └── example_rag_usage.py          # Примеры работы с RAG системой
│
├── 📁 tests/                          # Тесты
│   └── test_pipeline.py              # Тесты пайплайна обработки
│
├── 📁 docs/                           # Документация
│   ├── QUICKSTART.md                 # Быстрый старт
│   ├── README_RAG.md                 # Документация базового преобразования
│   ├── README_RAG_FULL.md            # Полная документация пайплайна
│   └── PROJECT_OVERVIEW.md           # Обзор проекта
│
├── 📁 config/                         # Конфигурационные файлы
│   └── config.example.json           # Пример конфигурации (создать при необходимости)
│
├── 📁 logs/                           # Логи работы системы
│   └── .gitkeep
│
├── 📄 requirements.txt                # Основные зависимости
├── 📄 requirements_rag.txt            # Зависимости для RAG системы
├── 📄 README.md                       # Главный README проекта
├── 📄 PROJECT_STRUCTURE.md            # Этот файл (описание структуры)
└── 📄 .gitignore                      # Git ignore файл
```

## Описание папок

### 📁 `data/`
Содержит все данные проекта:
- **`raw/`** - исходные файлы (Markdown, текстовые файлы)
- **`processed/`** - обработанные данные в структурированных форматах
- **`output/`** - результаты работы скриптов

### 📁 `rag_pipeline/`
Модули RAG системы:
- **`data_cleaner.py`** - очистка и нормализация данных
- **`text_chunker.py`** - разбиение длинных текстов на чанки
- **`embeddings.py`** - генерация векторных эмбеддингов
- **`vector_db.py`** - работа с векторными БД (Pinecone, Qdrant)
- **`rag_system.py`** - интеграция поиска и генерации ответов

### 📁 `scripts/`
Основные скрипты для работы:
- **`markdown_to_rag.py`** - преобразование Markdown таблиц в JSON/JSONL
- **`rag_pipeline_full.py`** - полный пайплайн обработки данных

### 📁 `examples/`
Примеры использования системы для обучения и вдохновения

### 📁 `tests/`
Тесты для проверки работоспособности модулей

### 📁 `docs/`
Вся документация проекта

### 📁 `config/`
Конфигурационные файлы (если нужны)

### 📁 `logs/`
Логи работы системы (создаются автоматически)

## Рабочий процесс

### 1. Подготовка данных
```
data/raw/test_asks.md → scripts/markdown_to_rag.py → data/processed/rag_data.jsonl
```

### 2. Обработка данных
```
data/processed/rag_data.jsonl → scripts/rag_pipeline_full.py → data/output/rag_processed.json
```

### 3. Использование
```
examples/example_rag_usage.py использует модули из rag_pipeline/
```

## Рекомендации

- ✅ **Исходные данные** храните только в `data/raw/`
- ✅ **Обработанные данные** сохраняйте в `data/processed/`
- ✅ **Результаты работы** выводите в `data/output/`
- ✅ **Документацию** обновляйте в `docs/`
- ✅ **Новые модули** добавляйте в `rag_pipeline/`

## Создание структуры

Для создания структуры папок запустите:

```bash
python create_structure.py
```

Для организации существующих файлов:

```bash
python organize_files.py
```

## Важные файлы

- **`requirements_rag.txt`** - все зависимости для RAG системы
- **`QUICKSTART.md`** - начните отсюда для быстрого старта
- **`README_RAG_FULL.md`** - полная документация

