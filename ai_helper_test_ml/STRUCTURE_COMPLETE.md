# ✅ Структура проекта создана!

## Что было сделано

### 1. Создана структура папок:
- ✅ `data/raw/` - исходные данные
- ✅ `data/processed/` - обработанные данные
- ✅ `data/output/` - выходные файлы
- ✅ `scripts/` - скрипты обработки
- ✅ `examples/` - примеры использования
- ✅ `tests/` - тесты
- ✅ `docs/` - документация
- ✅ `config/` - конфигурация
- ✅ `logs/` - логи

### 2. Файлы организованы:
- ✅ Исходные данные → `data/raw/`
- ✅ Обработанные данные → `data/processed/`
- ✅ Скрипты → `scripts/`
- ✅ Примеры → `examples/`
- ✅ Тесты → `tests/`
- ✅ Документация → `docs/`

### 3. Обновлены пути:
- ✅ Пути по умолчанию в скриптах обновлены под новую структуру
- ✅ Создан `.gitignore` файл
- ✅ Создан главный `README.md`

## Текущая структура

```
ai_helper/
├── data/
│   ├── raw/              # Исходные данные
│   │   └── test_asks.md
│   ├── processed/        # Обработанные данные
│   │   ├── rag_data.json
│   │   ├── rag_data.jsonl
│   │   └── rag_data.csv
│   └── output/           # Выходные файлы
│
├── rag_pipeline/         # Модули RAG системы
│   ├── data_cleaner.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_db.py
│   └── rag_system.py
│
├── scripts/              # Скрипты
│   ├── markdown_to_rag.py
│   └── rag_pipeline_full.py
│
├── examples/             # Примеры
│   └── example_rag_usage.py
│
├── tests/                # Тесты
│   └── test_pipeline.py
│
├── docs/                 # Документация
│   ├── QUICKSTART.md
│   ├── README_RAG.md
│   ├── README_RAG_FULL.md
│   └── PROJECT_OVERVIEW.md
│
├── config/               # Конфигурация
├── logs/                 # Логи
│
├── README.md             # Главный README
├── PROJECT_STRUCTURE.md  # Описание структуры
├── requirements_rag.txt  # Зависимости
└── .gitignore           # Git ignore
```

## Как использовать

### Преобразование Markdown
```bash
python scripts/markdown_to_rag.py --input data/raw/test_asks.md --output data/processed/rag_data
```

### Полная обработка
```bash
python scripts/rag_pipeline_full.py --input data/processed/rag_data.jsonl --output data/output/processed.json
```

### Тестирование
```bash
python tests/test_pipeline.py
```

## Документация

- **[README.md](README.md)** - главная страница проекта
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - быстрый старт
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - детальное описание структуры
- **[docs/README_RAG_FULL.md](docs/README_RAG_FULL.md)** - полная документация

## Дополнительно

Скрипты для работы со структурой:
- `create_structure.py` - создание структуры папок
- `organize_files.py` - организация файлов по структуре

---

**Проект готов к работе!** 🚀

