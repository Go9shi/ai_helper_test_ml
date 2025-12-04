# Полный пайплайн RAG системы

Комплексное решение для подготовки данных и работы с RAG (Retrieval-Augmented Generation) системой.

## Возможности

✅ **Очистка данных** - удаление дубликатов, нормализация, исправление опечаток  
✅ **Разбиение на чанки** - автоматическое разбиение длинных текстов на смысловые фрагменты  
✅ **Генерация эмбеддингов** - поддержка OpenAI и Sentence Transformers  
✅ **Векторные БД** - интеграция с Pinecone, Qdrant  
✅ **Поиск и генерация** - семантический поиск + генерация ответов через LLM  

## Установка

### 1. Установите зависимости

```bash
pip install -r requirements_rag.txt
```

### 2. Настройте переменные окружения (опционально)

Для OpenAI эмбеддингов и LLM:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Для Pinecone:
```bash
export PINECONE_API_KEY="your-pinecone-api-key"
```

Для Anthropic Claude (опционально):
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Быстрый старт

### Шаг 1: Подготовка данных

Сначала преобразуйте Markdown в структурированный формат:

```bash
python markdown_to_rag.py --input test_asks.md --output rag_data.jsonl
```

### Шаг 2: Полная обработка данных

Запустите полный пайплайн обработки:

```bash
# Базовый запуск (только обработка, без загрузки в БД)
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --output rag_processed.json \
    --embedding-provider sentence_transformers \
    --chunk-size 500 \
    --chunk-overlap 50
```

### Шаг 3: Загрузка в векторную БД

#### Вариант A: Pinecone

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --output rag_processed.json \
    --embedding-provider sentence_transformers \
    --upload-db \
    --db-provider pinecone \
    --pinecone-api-key "your-key" \
    --pinecone-index "rag-index"
```

#### Вариант B: Qdrant

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --output rag_processed.json \
    --embedding-provider sentence_transformers \
    --upload-db \
    --db-provider qdrant \
    --qdrant-url "http://localhost:6333" \
    --qdrant-collection "rag_collection"
```

### Шаг 4: Использование RAG системы

Создайте скрипт для поиска и генерации ответов (см. `example_rag_usage.py`):

```python
from rag_pipeline.rag_system import RAGSystem
from rag_pipeline.embeddings import create_embedding_generator
from rag_pipeline.vector_db import create_vector_db

# Инициализация компонентов
vector_db = create_vector_db(
    provider="pinecone",
    api_key="your-key",
    index_name="rag-index"
)

embedding_generator = create_embedding_generator(
    provider="sentence_transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

rag = RAGSystem(
    vector_db=vector_db,
    embedding_generator=embedding_generator,
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Поиск и генерация ответа
result = rag.ask("Как проверить баланс по карте?", top_k=5)
print(result['answer'])
```

## Детальное описание модулей

### 1. Очистка данных (`rag_pipeline/data_cleaner.py`)

```python
from rag_pipeline.data_cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_records = cleaner.clean_records(records, remove_duplicates=True)

# Статистика
stats = cleaner.get_statistics(cleaned_records)
print(f"Категорий: {len(stats['categories'])}")
```

**Возможности:**
- Удаление дубликатов
- Нормализация текста (удаление лишних пробелов, нормализация знаков препинания)
- Унификация формата вопросов и ответов

### 2. Разбиение на чанки (`rag_pipeline/text_chunker.py`)

```python
from rag_pipeline.text_chunker import TextChunker

chunker = TextChunker(
    chunk_size=500,      # Максимальный размер чанка
    chunk_overlap=50     # Перекрытие между чанками
)

chunked_records = chunker.chunk_records(records, enable_chunking=True)
```

**Особенности:**
- Разбиение по предложениям
- Сохранение смысловой целостности
- Перекрытие между чанками для контекста

### 3. Генерация эмбеддингов (`rag_pipeline/embeddings.py`)

#### OpenAI (платно, высокое качество)

```python
from rag_pipeline.embeddings import create_embedding_generator

generator = create_embedding_generator(
    provider="openai",
    model="text-embedding-3-small",
    api_key="your-key"
)
```

#### Sentence Transformers (бесплатно, локально)

```python
generator = create_embedding_generator(
    provider="sentence_transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Мультиязычная модель (без Xet Storage)
)
```

**Рекомендуемые модели:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - мультиязычная, быстрая, без Xet Storage (рекомендуется)
- `sentence-transformers/distiluse-base-multilingual-cased` - мультиязычная альтернатива
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - более точная мультиязычная модель
- `intfloat/multilingual-e5-base` - может использовать Xet Storage (может вызывать проблемы с загрузкой)

### 4. Векторные БД (`rag_pipeline/vector_db.py`)

#### Pinecone (облачный сервис)

```python
from rag_pipeline.vector_db import create_vector_db

db = create_vector_db(
    provider="pinecone",
    api_key="your-key",
    index_name="rag-index"
)
```

#### Qdrant (самостоятельный хостинг)

```python
db = create_vector_db(
    provider="qdrant",
    url="http://localhost:6333",
    collection_name="rag_collection"
)
```

### 5. RAG система (`rag_pipeline/rag_system.py`)

Полная интеграция поиска и генерации:

```python
from rag_pipeline.rag_system import RAGSystem

rag = RAGSystem(
    vector_db=vector_db,
    embedding_generator=embedding_generator,
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Полный цикл: поиск + генерация
result = rag.ask("Ваш вопрос", top_k=5)

# Только поиск
chunks = rag.search_relevant_chunks("Ваш вопрос", top_k=5)

# Поиск с фильтрацией
result = rag.ask(
    "Ваш вопрос",
    top_k=5,
    filter_dict={"category": "Безопасность"}
)
```

## Параметры командной строки

### `rag_pipeline_full.py`

```
--input, -i              Входной JSON/JSONL файл (обязательно)
--output, -o             Выходной JSON файл (по умолчанию: rag_processed.json)
--no-cleaning            Отключить очистку данных
--no-chunking            Отключить разбиение на чанки
--chunk-size             Размер чанка (по умолчанию: 500)
--chunk-overlap          Перекрытие между чанками (по умолчанию: 50)
--embedding-provider     Провайдер ('openai' или 'sentence_transformers')
--embedding-model        Модель для эмбеддингов
--no-save-embeddings     Не сохранять эмбеддинги в файл
--upload-db              Загрузить в векторную БД
--db-provider            Провайдер БД ('pinecone' или 'qdrant')
--pinecone-api-key       API ключ Pinecone
--pinecone-index         Название индекса Pinecone
--qdrant-url             URL Qdrant сервера
--qdrant-collection      Название коллекции Qdrant
```

## Примеры использования

### Пример 1: Полная обработка с загрузкой в Pinecone

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --embedding-provider sentence_transformers \
    --embedding-model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --chunk-size 500 \
    --upload-db \
    --db-provider pinecone \
    --pinecone-api-key "$PINECONE_API_KEY" \
    --pinecone-index "bank-faq"
```

### Пример 2: Только обработка без БД

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --output processed.json \
    --embedding-provider openai \
    --embedding-model "text-embedding-3-small" \
    --no-save-embeddings
```

### Пример 3: Без разбиения на чанки

```bash
python rag_pipeline_full.py \
    --input rag_data.jsonl \
    --no-chunking \
    --embedding-provider sentence_transformers
```

## Структура проекта

```
ai_helper/
├── markdown_to_rag.py          # Преобразование Markdown → JSON/JSONL
├── rag_pipeline_full.py        # Полный пайплайн обработки
├── example_rag_usage.py        # Примеры использования RAG
├── requirements_rag.txt        # Зависимости
├── rag_pipeline/
│   ├── __init__.py
│   ├── data_cleaner.py         # Очистка данных
│   ├── text_chunker.py         # Разбиение на чанки
│   ├── embeddings.py           # Генерация эмбеддингов
│   ├── vector_db.py            # Работа с векторными БД
│   └── rag_system.py           # RAG система
└── README_RAG_FULL.md          # Эта документация
```

## Рекомендации

### Выбор модели эмбеддингов

- **OpenAI** - если нужна максимальная точность и есть бюджет
- **Sentence Transformers** - для бесплатного локального использования
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - хороший баланс качества и скорости (рекомендуется)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - выше качество, медленнее
  - `sentence-transformers/distiluse-base-multilingual-cased` - альтернативная мультиязычная модель

### Размер чанков

- **200-300 символов** - для коротких FAQ
- **500-800 символов** - стандартный размер
- **1000+ символов** - для длинных документов

### Перекрытие чанков

- **10-20% от размера чанка** - стандартное значение
- Увеличивайте для сохранения контекста между чанками

## Troubleshooting

### Ошибка при загрузке модели Sentence Transformers

```bash
# Установите дополнительные зависимости
pip install torch torchvision torchaudio
```

### Проблемы с кодировкой

Убедитесь, что все файлы в UTF-8:

```python
# При сохранении
with open('file.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
```

### Ошибка подключения к Pinecone

Проверьте:
1. Правильность API ключа
2. Существование индекса
3. Размерность векторов (должна совпадать с моделью)

## Дальнейшее развитие

- [ ] Поддержка PostgreSQL с pgvector
- [ ] Веб-интерфейс для RAG системы
- [ ] Метрики качества поиска (precision, recall)
- [ ] Fine-tuning моделей на своих данных
- [ ] Поддержка других LLM (Llama, Mistral)

