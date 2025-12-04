"""
Пример использования RAG системы для поиска и генерации ответов.
"""

import os
from rag_pipeline.rag_system import RAGSystem
from rag_pipeline.embeddings import create_embedding_generator
from rag_pipeline.vector_db import create_vector_db


def example_search_and_answer():
    """Пример использования RAG системы для поиска и генерации ответов."""
    
    # 1. Настройка подключения к векторной БД
    print("[INFO] Подключение к векторной БД...")
    
    # Пример для Pinecone
    vector_db = create_vector_db(
        provider="pinecone",
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="rag-index"  # Замените на имя вашего индекса
    )
    
    # Или для Qdrant:
    # vector_db = create_vector_db(
    #     provider="qdrant",
    #     url="http://localhost:6333",
    #     collection_name="rag_collection"
    # )
    
    # 2. Настройка генератора эмбеддингов
    print("[INFO] Инициализация генератора эмбеддингов...")
    
    # Для OpenAI (требует API ключ)
    # embedding_generator = create_embedding_generator(
    #     provider="openai",
    #     model="text-embedding-3-small",
    #     api_key=os.getenv("OPENAI_API_KEY")
    # )
    
    # Для Sentence Transformers (локально, бесплатно)
    embedding_generator = create_embedding_generator(
        provider="sentence_transformers",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 3. Создание RAG системы
    print("[INFO] Создание RAG системы...")
    rag = RAGSystem(
        vector_db=vector_db,
        embedding_generator=embedding_generator,
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        llm_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 4. Примеры запросов
    queries = [
        "Как проверить баланс по карте?",
        "Что делать при утере карты?",
        "Как перевести деньги по номеру телефона?",
        "Какие комиссии за переводы?",
    ]
    
    print("\n" + "=" * 60)
    print("Примеры использования RAG системы:")
    print("=" * 60)
    
    for query in queries:
        print(f"\n[QUERY] {query}")
        print("-" * 60)
        
        # Поиск релевантных фрагментов
        chunks = rag.search_relevant_chunks(query, top_k=3)
        print(f"\n[SEARCH] Найдено {len(chunks)} релевантных фрагментов:")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  Фрагмент {i} (релевантность: {chunk.get('score', 0):.3f}):")
            print(f"    {chunk.get('text', chunk.get('answer', ''))[:200]}...")
        
        # Генерация ответа
        result = rag.ask(query, top_k=3)
        
        print(f"\n[ANSWER] {result['answer']}")
        print("-" * 60)
    
    # 5. Поиск с фильтрацией
    print("\n" + "=" * 60)
    print("Пример поиска с фильтрацией по категории:")
    print("=" * 60)
    
    query = "Как пополнить карту?"
    filter_dict = {"category": "Общие операции"}
    
    result = rag.ask(
        query,
        top_k=3,
        filter_dict=filter_dict
    )
    
    print(f"\n[QUERY] {query}")
    print(f"[FILTER] Категория: {filter_dict['category']}")
    print(f"[ANSWER] {result['answer']}")


def example_search_only():
    """Пример только поиска без генерации ответа (без LLM)."""
    
    print("[INFO] Инициализация компонентов для поиска...")
    
    # Векторная БД
    vector_db = create_vector_db(
        provider="pinecone",
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="rag-index"
    )
    
    # Генератор эмбеддингов
    embedding_generator = create_embedding_generator(
        provider="sentence_transformers",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Поиск релевантных фрагментов
    query = "Как заблокировать карту?"
    
    # Генерируем эмбеддинг запроса
    query_embedding = embedding_generator.generate([query])[0]
    
    # Ищем в БД
    results = vector_db.search(
        query_vector=query_embedding,
        top_k=5
    )
    
    print(f"\n[QUERY] {query}")
    print(f"[RESULTS] Найдено {len(results)} результатов:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Релевантность: {result.get('score', 0):.3f}")
        print(f"   Вопрос: {result.get('question', 'N/A')}")
        print(f"   Ответ: {result.get('answer', 'N/A')[:150]}...")
        print(f"   Категория: {result.get('category', 'N/A')}")
        print()


if __name__ == "__main__":
    # Запустите нужный пример
    
    # Пример полного использования (с LLM)
    # example_search_and_answer()
    
    # Пример только поиска (без LLM)
    example_search_only()

