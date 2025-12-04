# Преобразование Markdown в формат для RAG

Этот скрипт преобразует Markdown файлы с таблицами в структурированный формат для загрузки в векторную базу данных RAG (Retrieval-Augmented Generation).

## Возможности

- Парсинг Markdown таблиц
- Извлечение структурированных данных (ID, вопрос, категория, ответ)
- Сохранение в нескольких форматах:
  - **JSON** - для просмотра и отладки
  - **JSONL** - для пакетной загрузки в векторные БД (Pinecone, Weaviate, Qdrant и др.)
  - **CSV** - для анализа в Excel/Google Sheets

## Структура выходных данных

Каждая запись содержит:

```json
{
  "id": "1",
  "question": "Как проверить баланс по карте?",
  "category": "Общие операции",
  "answer": "Проверьте в мобильном приложении...",
  "section": "Общие операции",
  "text": "Вопрос: ...\nОтвет: ...",
  "metadata": {
    "section": "Общие операции",
    "category": "Общие операции",
    "source": "test_asks.md"
  }
}
```

### Поля:

- **id** - уникальный идентификатор записи
- **question** - вопрос пользователя
- **category** - категория вопроса
- **answer** - ответ на вопрос
- **section** - раздел из Markdown (заголовок ##)
- **text** - объединенный текст вопроса и ответа для создания эмбеддингов
- **metadata** - дополнительные метаданные для фильтрации в векторной БД

## Использование

### Базовое использование

```bash
python markdown_to_rag.py
```

Скрипт автоматически:
1. Читает файл `test_asks.md`
2. Парсит все таблицы
3. Создает три выходных файла:
   - `rag_data.json`
   - `rag_data.jsonl`
   - `rag_data.csv`

### Параметры командной строки

```bash
# Указать входной файл
python markdown_to_rag.py --input my_file.md

# Указать имя выходных файлов
python markdown_to_rag.py --output my_output

# Сохранить только в JSONL формат
python markdown_to_rag.py --format jsonl

# Комбинирование параметров
python markdown_to_rag.py -i test_asks.md -o output -f jsonl
```

**Параметры:**
- `--input, -i` - входной Markdown файл (по умолчанию: `test_asks.md`)
- `--output, -o` - базовое имя выходных файлов без расширения (по умолчанию: `rag_data`)
- `--format` - формат выходных файлов: `json`, `jsonl`, `csv` или `all` (по умолчанию: `all`)

## Загрузка в векторную БД

### Пример для Pinecone

```python
import json
from pinecone import Pinecone

# Читаем JSONL файл
with open('rag_data.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

# Подготавливаем данные для Pinecone
vectors = []
for record in records:
    # Здесь нужно создать эмбеддинг для record['text']
    # используя OpenAI, Sentence Transformers и т.д.
    embedding = create_embedding(record['text'])
    
    vectors.append({
        'id': record['id'],
        'values': embedding,
        'metadata': record['metadata']
    })

# Загружаем в Pinecone
pc = Pinecone(api_key="your-api-key")
index = pc.Index("your-index-name")
index.upsert(vectors=vectors)
```

### Пример для Qdrant

```python
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

with open('rag_data.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

points = []
for record in records:
    embedding = create_embedding(record['text'])
    points.append(
        PointStruct(
            id=int(record['id']),
            vector=embedding,
            payload={
                'question': record['question'],
                'answer': record['answer'],
                'category': record['category'],
                'section': record['section']
            }
        )
    )

client.upsert(
    collection_name="rag_collection",
    points=points
)
```

## Требования

- Python 3.7+
- pandas (для CSV экспорта)

Установка зависимостей:

```bash
pip install pandas
```

## Формат входного Markdown

Скрипт ожидает Markdown файл со следующей структурой:

```markdown
## Название раздела
| ID | Вопрос | Категория | Пример ответа |
|----|--------|-----------|--------------|
| 1 | Вопрос 1? | Категория 1 | Ответ 1 |
| 2 | Вопрос 2? | Категория 2 | Ответ 2 |
```

## Примечания

- Скрипт автоматически определяет разделы по заголовкам `##`
- Каждая строка таблицы становится отдельной записью
- Поле `text` объединяет вопрос и ответ для лучшего поиска в RAG системе

