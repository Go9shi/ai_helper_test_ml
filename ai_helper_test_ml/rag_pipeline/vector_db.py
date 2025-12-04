"""
Модуль для работы с векторными базами данных.
Поддерживает Pinecone, Qdrant и PostgreSQL с pgvector.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class VectorDB(ABC):
    """Базовый класс для работы с векторной БД."""
    
    @abstractmethod
    def upsert(self, records: List[Dict[str, Any]]) -> bool:
        """Загружает записи в векторную БД."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Выполняет поиск по векторной БД."""
        pass
    
    @abstractmethod
    def delete_all(self) -> bool:
        """Удаляет все записи из векторной БД."""
        pass


class PineconeVectorDB(VectorDB):
    """Реализация для Pinecone."""
    
    def __init__(self, api_key: str, index_name: str, environment: Optional[str] = None):
        """
        Инициализация Pinecone клиента.
        
        Args:
            api_key: API ключ Pinecone
            index_name: Название индекса
            environment: Окружение (для старой версии API)
        """
        try:
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            self.index = self.pc.Index(index_name)
            print(f"[INFO] Подключено к Pinecone индексу: {index_name}")
        except ImportError:
            raise ImportError("Для использования PineconeVectorDB установите: pip install pinecone-client")
        except Exception as e:
            raise Exception(f"Ошибка при подключении к Pinecone: {e}")
    
    def upsert(self, records: List[Dict[str, Any]], 
               embedding_field: str = 'embedding',
               id_field: str = 'id',
               batch_size: int = 100) -> bool:
        """
        Загружает записи в Pinecone.
        
        Args:
            records: Список записей
            embedding_field: Поле с эмбеддингом
            id_field: Поле с ID
            batch_size: Размер батча для загрузки
            
        Returns:
            True если успешно
        """
        try:
            vectors_to_upsert = []
            
            for record in records:
                record_id = str(record.get(id_field, ''))
                embedding = record.get(embedding_field, [])
                
                if not embedding:
                    continue
                
                # Метаданные для Pinecone (без эмбеддинга)
                metadata = {
                    'text': record.get('text', ''),
                    'question': record.get('question', ''),
                    'answer': record.get('answer', ''),
                    'category': record.get('category', ''),
                    'section': record.get('section', ''),
                }
                
                # Добавляем дополнительные метаданные
                if 'metadata' in record:
                    for k, v in record['metadata'].items():
                        if isinstance(v, (str, int, float, bool)):
                            metadata[k] = v
                
                vectors_to_upsert.append({
                    'id': record_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Загружаем батчами
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"[INFO] Загружено {min(i + batch_size, len(vectors_to_upsert))}/{len(vectors_to_upsert)} записей")
            
            print(f"[SUCCESS] Всего загружено {len(vectors_to_upsert)} записей в Pinecone")
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке в Pinecone: {e}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск в Pinecone.
        
        Args:
            query_vector: Вектор запроса
            top_k: Количество результатов
            filter_dict: Фильтр по метаданным
            
        Returns:
            Список результатов поиска
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    **match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            print(f"[ERROR] Ошибка при поиске в Pinecone: {e}")
            return []
    
    def delete_all(self) -> bool:
        """Удаляет все записи из Pinecone."""
        try:
            # Удаляем все векторы
            self.index.delete(delete_all=True)
            print("[INFO] Все записи удалены из Pinecone")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка при удалении из Pinecone: {e}")
            return False


class QdrantVectorDB(VectorDB):
    """Реализация для Qdrant."""
    
    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        """
        Инициализация Qdrant клиента.
        
        Args:
            url: URL Qdrant сервера
            collection_name: Название коллекции
            api_key: API ключ (если требуется)
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = collection_name
            
            # Проверяем существование коллекции
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                print(f"[WARNING] Коллекция {collection_name} не найдена. Создайте её вручную или через API.")
            
            print(f"[INFO] Подключено к Qdrant коллекции: {collection_name}")
        except ImportError:
            raise ImportError("Для использования QdrantVectorDB установите: pip install qdrant-client")
    
    def upsert(self, records: List[Dict[str, Any]],
               embedding_field: str = 'embedding',
               id_field: str = 'id',
               vector_size: Optional[int] = None) -> bool:
        """
        Загружает записи в Qdrant.
        
        Args:
            records: Список записей
            embedding_field: Поле с эмбеддингом
            id_field: Поле с ID
            vector_size: Размер вектора (определится автоматически если None)
            
        Returns:
            True если успешно
        """
        try:
            from qdrant_client.models import PointStruct
            
            points = []
            
            for record in records:
                record_id = record.get(id_field, '')
                embedding = record.get(embedding_field, [])
                
                if not embedding:
                    continue
                
                # Определяем размер вектора
                if vector_size is None:
                    vector_size = len(embedding)
                
                # Payload (метаданные)
                payload = {
                    'text': record.get('text', ''),
                    'question': record.get('question', ''),
                    'answer': record.get('answer', ''),
                    'category': record.get('category', ''),
                    'section': record.get('section', ''),
                }
                
                # Добавляем дополнительные метаданные
                if 'metadata' in record:
                    for k, v in record['metadata'].items():
                        if isinstance(v, (str, int, float, bool)):
                            payload[k] = v
                
                # Qdrant требует числовой ID или строку
                point_id = int(record_id) if record_id.isdigit() else hash(record_id) % (2**63)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )
            
            # Загружаем точки
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"[SUCCESS] Загружено {len(points)} записей в Qdrant")
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке в Qdrant: {e}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Выполняет поиск в Qdrant.
        
        Args:
            query_vector: Вектор запроса
            top_k: Количество результатов
            filter_dict: Фильтр по payload
            
        Returns:
            Список результатов поиска
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_dict
            )
            
            return [
                {
                    'id': hit.id,
                    'score': hit.score,
                    **hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            print(f"[ERROR] Ошибка при поиске в Qdrant: {e}")
            return []
    
    def delete_all(self) -> bool:
        """Удаляет все записи из Qdrant."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"all": True}
            )
            print("[INFO] Все записи удалены из Qdrant")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка при удалении из Qdrant: {e}")
            return False


def create_vector_db(provider: str = "pinecone", **kwargs) -> VectorDB:
    """
    Фабричная функция для создания подключения к векторной БД.
    
    Args:
        provider: Провайдер ('pinecone' или 'qdrant')
        **kwargs: Параметры для инициализации БД
        
    Returns:
        Экземпляр VectorDB
    """
    provider = provider.lower()
    
    if provider == "pinecone":
        return PineconeVectorDB(**kwargs)
    elif provider == "qdrant":
        return QdrantVectorDB(**kwargs)
    else:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}. Используйте 'pinecone' или 'qdrant'")

