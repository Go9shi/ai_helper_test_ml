# Решение проблем (Troubleshooting)

## Проблемы с загрузкой моделей из Hugging Face

### Проблема: Таймауты при загрузке модели (Xet Storage)

Если вы видите ошибки типа:
```
HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Read timed out.
```

Это означает, что Hugging Face пытается использовать Xet Storage, но соединение нестабильно.

### Решение 1: Использовать скрипт запуска (рекомендуется)

**Windows:**
```bash
tests\run_test_embeddings.bat
```

**Linux/Mac:**
```bash
chmod +x tests/run_test_embeddings.sh
./tests/run_test_embeddings.sh
```

### Решение 2: Установить переменные окружения вручную

**Windows PowerShell:**
```powershell
$env:HF_HUB_DISABLE_XET="1"
$env:HF_HUB_ENABLE_XET="0"
$env:HF_HUB_DOWNLOAD_TIMEOUT="1200"
$env:HF_DOWNLOAD_MAX_RETRIES="5"
python tests/test_embeddings.py
```

**Windows CMD:**
```cmd
set HF_HUB_DISABLE_XET=1
set HF_HUB_ENABLE_XET=0
set HF_HUB_DOWNLOAD_TIMEOUT=1200
set HF_DOWNLOAD_MAX_RETRIES=5
python tests/test_embeddings.py
```

**Linux/Mac:**
```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_XET=0
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export HF_DOWNLOAD_MAX_RETRIES=5
python tests/test_embeddings.py
```

### Решение 3: Использовать модель без Xet Storage

По умолчанию используется модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, которая не должна использовать Xet Storage. Если проблема сохраняется, попробуйте другую модель:

```python
from rag_pipeline.embeddings import create_embedding_generator

generator = create_embedding_generator(
    provider="sentence_transformers",
    model_name="sentence-transformers/distiluse-base-multilingual-cased"
)
```

### Решение 4: Увеличить таймауты

Если загрузка прерывается из-за медленного интернета:

```python
generator = create_embedding_generator(
    provider="sentence_transformers",
    download_timeout=1800,  # 30 минут
    max_retries=5,
    retry_delay=10.0
)
```

### Решение 5: Очистить кэш Hugging Face

Иногда проблемы могут быть связаны с поврежденным кэшем:

**Windows:**
```cmd
rmdir /s /q %USERPROFILE%\.cache\huggingface
```

**Linux/Mac:**
```bash
rm -rf ~/.cache/huggingface
```

### Решение 6: Загрузить модель вручную

1. Перейдите на https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
2. Скачайте файлы модели вручную
3. Поместите их в папку кэша: `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`

## Другие проблемы

### Проблема: Модель не загружается после нескольких попыток

**Решение:** Увеличьте количество попыток и таймаут:
```python
generator = create_embedding_generator(
    provider="sentence_transformers",
    download_timeout=1800,
    max_retries=10,
    retry_delay=15.0
)
```

### Проблема: Недостаточно памяти

**Решение:** Используйте модель меньшего размера или обрабатывайте тексты батчами меньшего размера:
```python
generator = create_embedding_generator(
    provider="sentence_transformers",
    batch_size=16  # Уменьшите размер батча
)
```

### Проблема: Медленная генерация эмбеддингов

**Решение:** 
- Используйте GPU, если доступно: `device="cuda"`
- Увеличьте размер батча: `batch_size=64`
- Используйте более быструю модель

## Получение помощи

Если проблема не решается:
1. Проверьте логи на наличие дополнительной информации
2. Убедитесь, что у вас установлены последние версии библиотек:
   ```bash
   pip install --upgrade sentence-transformers huggingface_hub
   ```
3. Проверьте подключение к интернету
4. Попробуйте использовать другую модель

