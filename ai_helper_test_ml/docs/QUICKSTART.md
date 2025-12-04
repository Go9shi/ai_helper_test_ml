# Быстрый старт RAG системы

## Шаг 1: Преобразование Markdown в структурированный формат

```bash
python scripts/markdown_to_rag.py --input data/raw/test_asks.md --output data/processed/rag_data
```

## Шаг 2: Установка зависимостей

```bash
# Для базовой обработки (без векторных БД)
pip install pandas sentence-transformers numpy

# Для полного функционала (включая Pinecone, OpenAI)
pip install -r requirements_rag.txt
```

## Шаг 3: Полная обработка данных

### Вариант A: Только обработка (без загрузки в БД)

```bash
python scripts/rag_pipeline_full.py \
    --input data/processed/rag_data.jsonl \
    --output data/output/rag_processed.json \
    --embedding-provider sentence_transformers
```

### Вариант B: С загрузкой в Pinecone

```bash
# Установите переменную окружения
export PINECONE_API_KEY="your-api-key"

# Запустите пайплайн
python scripts/rag_pipeline_full.py \
    --input data/processed/rag_data.jsonl \
    --output data/output/rag_processed.json \
    --embedding-provider sentence_transformers \
    --upload-db \
    --db-provider pinecone \
    --pinecone-index "your-index-name"
```

### Вариант C: С загрузкой в Qdrant (локально)

```bash
# Запустите Qdrant локально (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Запустите пайплайн
python scripts/rag_pipeline_full.py \
    --input data/processed/rag_data.jsonl \
    --output data/output/rag_processed.json \
    --embedding-provider sentence_transformers \
    --upload-db \
    --db-provider qdrant \
    --qdrant-url "http://localhost:6333" \
    --qdrant-collection "rag_collection"
```

## Шаг 4: Использование RAG системы

Создайте скрипт `use_rag.py`:

```python
import os
from rag_pipeline.rag_system import RAGSystem
from rag_pipeline.embeddings import create_embedding_generator
from rag_pipeline.vector_db import create_vector_db

# Инициализация
vector_db = create_vector_db(
    provider="pinecone",  # или "qdrant"
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="your-index-name"
)

embedding_generator = create_embedding_generator(
    provider="sentence_transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

rag = RAGSystem(
    vector_db=vector_db,
    embedding_generator=embedding_generator,
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    llm_api_key=os.getenv("OPENAI_API_KEY")
)

# Использование
result = rag.ask("Как проверить баланс по карте?", top_k=5)
print(result['answer'])
```

Запустите:

```bash
export OPENAI_API_KEY="your-openai-key"
python use_rag.py
```

## Что происходит в пайплайне?

1. **Загрузка данных** из JSON/JSONL файла
2. **Очистка данных** - удаление дубликатов, нормализация
3. **Разбиение на чанки** - разбиение длинных текстов (опционально)
4. **Генерация эмбеддингов** - создание векторных представлений
5. **Загрузка в БД** - сохранение в векторную БД (опционально)

## Минимальный пример (без БД)

Если нужно просто обработать данные без загрузки в БД:

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --output processed.json \
    --embedding-provider sentence_transformers \
    --no-save-embeddings  # Эмбеддинги не сохраняются (файл будет меньше)
```

Результат: файл `processed.json` с очищенными данными и разбитыми на чанки.

## Дополнительная помощь

- Подробная документация: `README_RAG_FULL.md`
- Примеры использования: `example_rag_usage.py`
- Базовое преобразование Markdown: `README_RAG.md`

