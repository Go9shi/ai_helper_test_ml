"""
Модуль для разбиения длинных текстов на смысловые фрагменты (чанки) для RAG.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Класс для представления текстового чанка."""
    text: str
    chunk_id: str
    chunk_index: int
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TextChunker:
    """Класс для разбиения текстов на смысловые фрагменты."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Инициализация TextChunker.
        
        Args:
            chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Перекрытие между чанками в символах
            min_chunk_size: Минимальный размер чанка
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список предложений
        """
        # Регулярное выражение для разбиения на предложения
        # Учитывает точки, восклицательные и вопросительные знаки
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """
        Разбивает текст на абзацы.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список абзацев
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_chunks_from_text(self, text: str, 
                                chunk_id_prefix: str = "chunk",
                                parent_metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Создает чанки из текста, разбивая по предложениям.
        
        Args:
            text: Исходный текст
            chunk_id_prefix: Префикс для ID чанков
            parent_metadata: Метаданные родительской записи
            
        Returns:
            Список чанков
        """
        # Если текст короткий, возвращаем как один чанк
        if len(text) <= self.chunk_size:
            chunk = Chunk(
                text=text,
                chunk_id=f"{chunk_id_prefix}_0",
                chunk_index=0,
                metadata=parent_metadata.copy() if parent_metadata else {}
            )
            return [chunk]
        
        chunks = []
        sentences = self.split_by_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Если одно предложение больше чанка, разбиваем его
            if sentence_length > self.chunk_size:
                # Сначала сохраняем текущий чанк если есть
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=f"{chunk_id_prefix}_{chunk_index}",
                        chunk_index=chunk_index,
                        metadata=parent_metadata.copy() if parent_metadata else {}
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
                
                # Разбиваем большое предложение по словам
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    
                    if temp_length + word_length > self.chunk_size and temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_id=f"{chunk_id_prefix}_{chunk_index}",
                            chunk_index=chunk_index,
                            metadata=parent_metadata.copy() if parent_metadata else {}
                        ))
                        chunk_index += 1
                        # Начинаем новый чанк с перекрытием
                        overlap_sentences = temp_chunk[-self.chunk_overlap // 10:] if len(temp_chunk) > self.chunk_overlap // 10 else temp_chunk
                        temp_chunk = overlap_sentences + [word]
                        temp_length = sum(len(w) for w in temp_chunk) + len(temp_chunk) - 1
                    else:
                        temp_chunk.append(word)
                        temp_length += word_length
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length
            
            # Добавляем предложение к текущему чанку
            elif current_length + sentence_length + 1 <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_id=f"{chunk_id_prefix}_{chunk_index}",
                            chunk_index=chunk_index,
                            metadata=parent_metadata.copy() if parent_metadata else {}
                        ))
                        chunk_index += 1
                    
                    # Начинаем новый чанк с перекрытием
                    if self.chunk_overlap > 0 and len(current_chunk) > 1:
                        overlap_count = max(1, len(current_chunk) // 3)
                        current_chunk = current_chunk[-overlap_count:] + [sentence]
                        current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{chunk_id_prefix}_{chunk_index}",
                    chunk_index=chunk_index,
                    metadata=parent_metadata.copy() if parent_metadata else {}
                ))
        
        return chunks
    
    def chunk_record(self, record: Dict[str, Any], 
                    chunk_text_field: str = 'text',
                    enable_chunking: bool = True) -> List[Dict[str, Any]]:
        """
        Разбивает запись на чанки.
        
        Args:
            record: Исходная запись
            chunk_text_field: Поле с текстом для разбиения
            enable_chunking: Включить ли разбиение на чанки
            
        Returns:
            Список записей-чанков
        """
        text = record.get(chunk_text_field, '')
        record_id = record.get('id', 'unknown')
        
        # Если разбиение отключено или текст короткий, возвращаем как есть
        if not enable_chunking or len(text) <= self.chunk_size:
            return [record]
        
        # Создаем чанки
        parent_metadata = {
            'parent_id': record_id,
            'original_question': record.get('question', ''),
            'original_answer': record.get('answer', ''),
            'category': record.get('category', ''),
            'section': record.get('section', ''),
            'source': record.get('metadata', {}).get('source', ''),
        }
        
        chunks = self.create_chunks_from_text(
            text,
            chunk_id_prefix=f"{record_id}",
            parent_metadata=parent_metadata
        )
        
        # Преобразуем чанки в формат записей
        chunk_records = []
        for chunk in chunks:
            chunk_record = {
                'id': chunk.chunk_id,
                'parent_id': record_id,
                'question': record.get('question', ''),
                'answer': record.get('answer', ''),
                'category': record.get('category', ''),
                'section': record.get('section', ''),
                'text': chunk.text,
                'chunk_index': chunk.chunk_index,
                'is_chunk': True,
                'metadata': {
                    **chunk.metadata,
                    'chunk_index': chunk.chunk_index,
                    'is_chunk': True,
                }
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    def chunk_records(self, records: List[Dict[str, Any]],
                     enable_chunking: bool = True,
                     min_text_length: int = 500) -> List[Dict[str, Any]]:
        """
        Разбивает список записей на чанки.
        
        Args:
            records: Список записей
            enable_chunking: Включить ли разбиение на чанки
            min_text_length: Минимальная длина текста для разбиения на чанки
            
        Returns:
            Список записей (исходные + чанки)
        """
        chunked_records = []
        total_chunks = 0
        
        for record in records:
            text = record.get('text', '')
            
            # Разбиваем только длинные тексты
            if enable_chunking and len(text) > min_text_length:
                chunks = self.chunk_record(record, enable_chunking=True)
                chunked_records.extend(chunks)
                total_chunks += len(chunks) - 1  # -1 потому что один был исходной записью
            else:
                # Короткие записи оставляем как есть
                record['is_chunk'] = False
                if 'metadata' not in record:
                    record['metadata'] = {}
                record['metadata']['is_chunk'] = False
                chunked_records.append(record)
        
        if enable_chunking and total_chunks > 0:
            print(f"[INFO] Создано чанков: {total_chunks}")
            print(f"[INFO] Всего записей после чанкинга: {len(chunked_records)}")
        
        return chunked_records

