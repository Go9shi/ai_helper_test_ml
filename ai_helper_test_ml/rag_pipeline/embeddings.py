"""
Модуль для генерации векторных эмбеддингов для текстов.
Поддерживает OpenAI и Sentence Transformers.
"""

import os
from typing import List, Optional, Dict, Any
import time
import logging

# КРИТИЧНО: Отключаем Xet Storage ДО любых импортов huggingface_hub
# Это должно быть установлено до импорта sentence_transformers
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_XET'] = '0'

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Базовый класс для генерации эмбеддингов."""
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        raise NotImplementedError


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Генератор эмбеддингов с использованием OpenAI API."""
    
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 batch_size: int = 100,
                 delay_between_batches: float = 0.1):
        """
        Инициализация OpenAI генератора.
        
        Args:
            model: Модель OpenAI для эмбеддингов
            api_key: API ключ OpenAI (если None, берется из переменной окружения)
            batch_size: Размер батча для обработки
            delay_between_batches: Задержка между батчами (секунды)
        """
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        if not self.api_key:
            raise ValueError("Необходим API ключ OpenAI. Установите OPENAI_API_KEY или передайте api_key")
        
        # Импортируем OpenAI только если используется
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Для использования OpenAIEmbeddingGenerator установите openai: pip install openai")
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги через OpenAI API.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        all_embeddings = []
        
        # Обрабатываем батчами
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Задержка между батчами
                if i + self.batch_size < len(texts):
                    time.sleep(self.delay_between_batches)
            
            except Exception as e:
                print(f"[ERROR] Ошибка при генерации эмбеддингов для батча {i//self.batch_size + 1}: {e}")
                # В случае ошибки добавляем пустые векторы
                for _ in batch:
                    all_embeddings.append([])
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддингов."""
        # text-embedding-3-small: 1536
        # text-embedding-3-large: 3072
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Генератор эмбеддингов с использованием Sentence Transformers."""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu",
                 batch_size: int = 32,
                 download_timeout: int = 600,
                 max_retries: int = 3,
                 retry_delay: float = 5.0):
        """
        Инициализация Sentence Transformer генератора.
        
        Args:
            model_name: Название модели из Sentence Transformers
            device: Устройство для вычислений (cpu/cuda)
            batch_size: Размер батча для обработки
            download_timeout: Таймаут загрузки в секундах (по умолчанию 600)
            max_retries: Максимальное количество попыток загрузки
            retry_delay: Задержка между попытками в секундах
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.download_timeout = download_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Отключаем Xet Storage для избежания проблем с таймаутами
            # Используем обычную HTTP загрузку через huggingface.co
            # Правильная переменная окружения для отключения Xet Storage
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            
            # Настройка таймаутов для Hugging Face из переменных окружения или параметров
            timeout_from_env = os.getenv('HF_HUB_DOWNLOAD_TIMEOUT')
            if timeout_from_env:
                download_timeout = int(timeout_from_env)
            else:
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(download_timeout)
            
            # Увеличиваем таймаут чтения для больших файлов
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(max(download_timeout, 600))  # Минимум 10 минут
            
            # Настройка количества попыток из переменных окружения
            max_retries_from_env = os.getenv('HF_DOWNLOAD_MAX_RETRIES')
            if max_retries_from_env:
                self.max_retries = int(max_retries_from_env)
            
            # Настройка задержки между попытками из переменных окружения
            retry_delay_from_env = os.getenv('HF_DOWNLOAD_RETRY_DELAY')
            if retry_delay_from_env:
                self.retry_delay = float(retry_delay_from_env)
            
            # Пытаемся загрузить модель с повторными попытками
            self.model = self._load_model_with_retry(model_name, device)
            print(f"[INFO] Модель загружена успешно")
        except ImportError:
            raise ImportError("Для использования SentenceTransformerEmbeddingGenerator установите: pip install sentence-transformers")
    
    def _load_model_with_retry(self, model_name: str, device: str):
        """
        Загружает модель с повторными попытками при ошибках сети.
        
        Args:
            model_name: Название модели
            device: Устройство для вычислений
            
        Returns:
            Загруженная модель SentenceTransformer
        """
        # Убеждаемся, что Xet Storage отключен перед импортом
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['HF_HUB_ENABLE_XET'] = '0'
        
        # Дополнительные настройки для отключения Xet Storage
        os.environ['HF_HUB_DISABLE_XET_WARNING'] = '1'
        
        # Пытаемся отключить Xet Storage через huggingface_hub API
        try:
            import huggingface_hub
            # Отключаем Xet Storage через константы, если доступно
            if hasattr(huggingface_hub, 'constants'):
                if hasattr(huggingface_hub.constants, 'ENABLE_XET'):
                    huggingface_hub.constants.ENABLE_XET = False
                # Пытаемся установить через другие атрибуты
                if hasattr(huggingface_hub.constants, 'DISABLE_XET'):
                    huggingface_hub.constants.DISABLE_XET = True
            
            # Пытаемся использовать file_download с явным отключением Xet
            if hasattr(huggingface_hub, 'file_download'):
                # Сохраняем оригинальную функцию
                original_file_download = huggingface_hub.file_download
                
                def patched_file_download(*args, **kwargs):
                    # Принудительно отключаем Xet в параметрах
                    kwargs['use_xet'] = False
                    kwargs['use_xet_storage'] = False
                    return original_file_download(*args, **kwargs)
                
                # Патчим функцию (но это может не сработать, так как она уже импортирована)
                # huggingface_hub.file_download = patched_file_download
        except (ImportError, AttributeError):
            pass
        
        from sentence_transformers import SentenceTransformer
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"[INFO] Попытка {attempt}/{self.max_retries} загрузки модели {model_name}...")
                if attempt == 1:
                    print("[INFO] Используется обычная HTTP загрузка (Xet Storage отключен)")
                    print("[INFO] Если возникают проблемы, попробуйте установить переменные окружения:")
                    print("       export HF_HUB_DISABLE_XET=1")
                    print("       export HF_HUB_DOWNLOAD_TIMEOUT=1200")
                
                # Настройка параметров загрузки
                # Пытаемся передать параметры для отключения Xet через cache_folder
                model_kwargs = {
                    'device': device,
                }
                
                # Пытаемся использовать локальный кэш и принудительную загрузку через обычный HTTP
                try:
                    import huggingface_hub
                    # Устанавливаем параметры для file_download через переменные окружения
                    # Это должно заставить использовать обычный HTTP
                    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(self.download_timeout)
                    
                    # Пытаемся установить параметры для отключения Xet через конфигурацию
                    # Это более надежный способ
                    try:
                        # Используем snapshot_download напрямую с явным отключением Xet
                        # перед загрузкой через SentenceTransformer
                        print("[INFO] Предварительная загрузка модели через huggingface_hub...")
                        huggingface_hub.snapshot_download(
                            repo_id=model_name,
                            resume_download=True,
                            local_files_only=False,
                        )
                        print("[INFO] Модель загружена в кэш, инициализируем SentenceTransformer...")
                    except Exception as preload_error:
                        # Если предзагрузка не удалась, продолжаем обычным способом
                        logger.warning(f"Предзагрузка не удалась: {preload_error}, продолжаем обычным способом")
                except ImportError:
                    pass
                
                # Загружаем модель через SentenceTransformer
                # Если модель уже в кэше, это должно работать быстрее
                model = SentenceTransformer(model_name, **model_kwargs)
                
                print(f"[INFO] Модель успешно загружена с попытки {attempt}")
                return model
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Проверяем, является ли это ошибкой сети/таймаута
                is_network_error = any(keyword in error_msg for keyword in [
                    'timeout', 'timed out', 'connection', 'network', 
                    'read timed out', 'max retries', 'connection pool'
                ])
                
                if is_network_error and attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))  # Экспоненциальная задержка
                    print(f"[WARNING] Ошибка сети при загрузке модели: {e}")
                    print(f"[INFO] Повторная попытка через {wait_time:.1f} секунд...")
                    time.sleep(wait_time)
                else:
                    # Если это не ошибка сети или закончились попытки
                    if attempt >= self.max_retries:
                        error_msg = (
                            f"\n[ERROR] Не удалось загрузить модель '{model_name}' после {self.max_retries} попыток.\n"
                            f"Последняя ошибка: {last_error}\n\n"
                            f"Рекомендации:\n"
                            f"1. Проверьте подключение к интернету\n"
                            f"2. Увеличьте таймаут загрузки через переменную окружения:\n"
                            f"   export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 минут\n"
                            f"3. Увеличьте количество попыток:\n"
                            f"   export HF_DOWNLOAD_MAX_RETRIES=5\n"
                            f"4. Или передайте параметры при создании генератора:\n"
                            f"   download_timeout=600, max_retries=5\n"
                            f"5. Попробуйте загрузить модель вручную:\n"
                            f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\""
                        )
                        print(error_msg)
                        raise Exception(error_msg)
                    else:
                        # Если это другая ошибка, пробрасываем сразу
                        raise
        
        # Если мы здесь, значит все попытки исчерпаны
        error_msg = (
            f"Не удалось загрузить модель '{model_name}' после {self.max_retries} попыток. "
            f"Последняя ошибка: {last_error}"
        )
        raise Exception(error_msg)
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги через Sentence Transformers.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Преобразуем в список списков
            return embeddings.tolist()
        
        except Exception as e:
            print(f"[ERROR] Ошибка при генерации эмбеддингов: {e}")
            return [[]] * len(texts)
    
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддингов."""
        return self.model.get_sentence_embedding_dimension()


def create_embedding_generator(provider: str = "openai", **kwargs) -> EmbeddingGenerator:
    """
    Фабричная функция для создания генератора эмбеддингов.
    
    Args:
        provider: Провайдер ('openai' или 'sentence_transformers')
        **kwargs: Параметры для инициализации генератора
        
    Returns:
        Экземпляр генератора эмбеддингов
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIEmbeddingGenerator(**kwargs)
    elif provider == "sentence_transformers" or provider == "sentence-transformer":
        return SentenceTransformerEmbeddingGenerator(**kwargs)
    else:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}. Используйте 'openai' или 'sentence_transformers'")


def add_embeddings_to_records(records: List[Dict[str, Any]],
                              embedding_generator: EmbeddingGenerator,
                              text_field: str = 'text',
                              embedding_field: str = 'embedding') -> List[Dict[str, Any]]:
    """
    Добавляет эмбеддинги к записям.
    
    Args:
        records: Список записей
        embedding_generator: Генератор эмбеддингов
        text_field: Поле с текстом для эмбеддинга
        embedding_field: Поле для сохранения эмбеддинга
        
    Returns:
        Список записей с добавленными эмбеддингами
    """
    texts = [record.get(text_field, '') for record in records]
    
    print(f"[INFO] Генерация эмбеддингов для {len(texts)} текстов...")
    embeddings = embedding_generator.generate(texts)
    
    # Добавляем эмбеддинги к записям
    for record, embedding in zip(records, embeddings):
        record[embedding_field] = embedding
        if 'metadata' not in record:
            record['metadata'] = {}
        record['metadata']['has_embedding'] = True
        record['metadata']['embedding_dimension'] = len(embedding) if embedding else 0
    
    print(f"[INFO] Эмбеддинги добавлены к {len(records)} записям")
    
    return records

