"""
Модуль для интеграции RAG системы с языковой моделью.
Обеспечивает поиск релевантных фрагментов и генерацию ответов.
"""

from typing import List, Dict, Any, Optional
from .vector_db import VectorDB
from .embeddings import EmbeddingGenerator


class RAGSystem:
    """Система RAG для поиска и генерации ответов."""
    
    def __init__(self,
                 vector_db: VectorDB,
                 embedding_generator: EmbeddingGenerator,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-3.5-turbo",
                 llm_api_key: Optional[str] = None):
        """
        Инициализация RAG системы.
        
        Args:
            vector_db: Векторная БД
            embedding_generator: Генератор эмбеддингов
            llm_provider: Провайдер LLM ('openai' или 'anthropic')
            llm_model: Модель LLM
            llm_api_key: API ключ для LLM
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        
        # Инициализация LLM клиента
        if self.llm_provider == "openai":
            try:
                import os
                from openai import OpenAI
                api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("Необходим API ключ OpenAI")
                self.llm_client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Для использования OpenAI установите: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                import os
                from anthropic import Anthropic
                api_key = llm_api_key or os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("Необходим API ключ Anthropic")
                self.llm_client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Для использования Anthropic установите: pip install anthropic")
        else:
            raise ValueError(f"Неподдерживаемый провайдер LLM: {llm_provider}")
    
    def search_relevant_chunks(self,
                              query: str,
                              top_k: int = 5,
                              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ищет релевантные фрагменты для запроса.
        
        Args:
            query: Текст запроса
            top_k: Количество релевантных фрагментов
            filter_dict: Фильтр по метаданным
            
        Returns:
            Список релевантных фрагментов
        """
        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_generator.generate([query])[0]
        
        # Ищем в векторной БД
        results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def generate_answer(self,
                       query: str,
                       context_chunks: List[Dict[str, Any]],
                       system_prompt: Optional[str] = None) -> str:
        """
        Генерирует ответ на основе найденных фрагментов.
        
        Args:
            query: Вопрос пользователя
            context_chunks: Релевантные фрагменты из БД
            system_prompt: Системный промпт
            
        Returns:
            Сгенерированный ответ
        """
        # Формируем контекст из найденных фрагментов
        context = "\n\n".join([
            f"Фрагмент {i+1}:\n{chunk.get('text', chunk.get('answer', ''))}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Системный промпт по умолчанию
        if system_prompt is None:
            system_prompt = """Ты помощник, который отвечает на вопросы пользователей на основе предоставленного контекста.
Используй только информацию из контекста. Если в контексте нет ответа на вопрос, так и скажи.
Отвечай кратко и по существу на русском языке."""
        
        # Формируем промпт
        user_prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответ:"""
        
        # Генерируем ответ через LLM
        if self.llm_provider == "openai":
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Ошибка при генерации ответа: {e}"
        
        elif self.llm_provider == "anthropic":
            try:
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=500,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text.strip()
            except Exception as e:
                return f"Ошибка при генерации ответа: {e}"
    
    def ask(self,
            query: str,
            top_k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None,
            system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Полный цикл RAG: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество релевантных фрагментов
            filter_dict: Фильтр по метаданным
            system_prompt: Системный промпт
            
        Returns:
            Словарь с ответом и метаданными
        """
        # Ищем релевантные фрагменты
        chunks = self.search_relevant_chunks(query, top_k, filter_dict)
        
        # Генерируем ответ
        answer = self.generate_answer(query, chunks, system_prompt)
        
        return {
            'query': query,
            'answer': answer,
            'relevant_chunks': chunks,
            'num_chunks': len(chunks)
        }

