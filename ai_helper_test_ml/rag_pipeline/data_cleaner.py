"""
Модуль для очистки и нормализации данных перед загрузкой в RAG систему.
"""

import re
import unicodedata
from typing import List, Dict, Any, Set
from collections import defaultdict


class DataCleaner:
    """Класс для очистки данных от дубликатов, опечаток и нормализации формата."""
    
    def __init__(self):
        self.seen_texts: Set[str] = set()
        self.seen_questions: Set[str] = set()
    
    def normalize_text(self, text: str) -> str:
        """
        Нормализует текст: убирает лишние пробелы, приводит к единому формату.
        
        Args:
            text: Исходный текст
            
        Returns:
            Нормализованный текст
        """
        if not text:
            return ""
        
        # Нормализуем unicode символы (например, разные типы кавычек)
        text = unicodedata.normalize('NFKC', text)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем пробелы в начале и конце
        text = text.strip()
        
        # Нормализуем знаки препинания
        text = re.sub(r'\.{2,}', '.', text)  # Множественные точки
        text = re.sub(r'\?{2,}', '?', text)  # Множественные вопросительные знаки
        text = re.sub(r'!{2,}', '!', text)   # Множественные восклицательные знаки
        
        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Добавляем пробелы после знаков препинания если их нет
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
        
        return text
    
    def normalize_question(self, question: str) -> str:
        """
        Нормализует вопрос: унифицирует формат, исправляет общие опечатки.
        
        Args:
            question: Исходный вопрос
            
        Returns:
            Нормализованный вопрос
        """
        if not question:
            return ""
        
        question = self.normalize_text(question)
        
        # Убираем знак вопроса если он есть в начале или множественные в конце
        question = question.strip().rstrip('?')
        
        # Исправляем общие опечатки
        question = question.replace('как?', 'как')
        question = question.replace('что?', 'что')
        question = question.replace('где?', 'где')
        
        # Добавляем знак вопроса в конец если его нет
        if question and not question.endswith('?'):
            question += '?'
        
        # Первая буква заглавная
        if question:
            question = question[0].upper() + question[1:] if len(question) > 1 else question.upper()
        
        return question
    
    def normalize_answer(self, answer: str) -> str:
        """
        Нормализует ответ: унифицирует формат.
        
        Args:
            answer: Исходный ответ
            
        Returns:
            Нормализованный ответ
        """
        if not answer:
            return ""
        
        answer = self.normalize_text(answer)
        
        # Первая буква заглавная
        if answer:
            answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
        
        return answer
    
    def is_duplicate(self, record: Dict[str, Any], similarity_threshold: float = 0.9) -> bool:
        """
        Проверяет, является ли запись дубликатом.
        
        Args:
            record: Запись для проверки
            similarity_threshold: Порог схожести (0-1)
            
        Returns:
            True если запись является дубликатом
        """
        # Проверяем по тексту вопроса (точное совпадение после нормализации)
        normalized_question = self.normalize_question(record.get('question', ''))
        normalized_question_lower = normalized_question.lower()
        
        # Простая проверка на точное совпадение
        if normalized_question_lower in self.seen_questions:
            return True
        
        # Проверяем по комбинации вопроса и ответа
        text = record.get('text', '')
        normalized_text = self.normalize_text(text).lower()
        
        if normalized_text in self.seen_texts:
            return True
        
        return False
    
    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Очищает одну запись от опечаток и нормализует формат.
        
        Args:
            record: Исходная запись
            
        Returns:
            Очищенная запись
        """
        cleaned = record.copy()
        
        # Нормализуем поля
        if 'question' in cleaned:
            cleaned['question'] = self.normalize_question(cleaned['question'])
        
        if 'answer' in cleaned:
            cleaned['answer'] = self.normalize_answer(cleaned['answer'])
        
        # Пересоздаем text поле
        if 'question' in cleaned and 'answer' in cleaned:
            cleaned['text'] = f"Вопрос: {cleaned['question']}\nОтвет: {cleaned['answer']}"
        
        # Нормализуем категорию и раздел
        if 'category' in cleaned:
            cleaned['category'] = self.normalize_text(cleaned['category']).strip()
        
        if 'section' in cleaned:
            cleaned['section'] = self.normalize_text(cleaned['section']).strip()
        
        # Обновляем metadata
        if 'metadata' in cleaned:
            cleaned['metadata'] = cleaned['metadata'].copy()
            if 'category' in cleaned['metadata']:
                cleaned['metadata']['category'] = cleaned.get('category', '')
            if 'section' in cleaned['metadata']:
                cleaned['metadata']['section'] = cleaned.get('section', '')
        
        return cleaned
    
    def clean_records(self, records: List[Dict[str, Any]], 
                     remove_duplicates: bool = True,
                     similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Очищает список записей от дубликатов и нормализует формат.
        
        Args:
            records: Список записей для очистки
            remove_duplicates: Удалять ли дубликаты
            similarity_threshold: Порог схожести для определения дубликатов
            
        Returns:
            Очищенный список записей
        """
        self.seen_texts.clear()
        self.seen_questions.clear()
        
        cleaned_records = []
        duplicates_removed = 0
        
        for record in records:
            # Очищаем запись
            cleaned = self.clean_record(record)
            
            # Проверяем на дубликаты
            if remove_duplicates and self.is_duplicate(cleaned, similarity_threshold):
                duplicates_removed += 1
                continue
            
            # Добавляем в список уникальных
            normalized_question = self.normalize_question(cleaned.get('question', '')).lower()
            normalized_text = self.normalize_text(cleaned.get('text', '')).lower()
            
            self.seen_questions.add(normalized_question)
            self.seen_texts.add(normalized_text)
            
            cleaned_records.append(cleaned)
        
        if remove_duplicates:
            print(f"[INFO] Удалено дубликатов: {duplicates_removed}")
        
        return cleaned_records
    
    def get_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Получает статистику по записям.
        
        Args:
            records: Список записей
            
        Returns:
            Словарь со статистикой
        """
        stats = {
            'total': len(records),
            'categories': defaultdict(int),
            'sections': defaultdict(int),
            'avg_question_length': 0,
            'avg_answer_length': 0,
        }
        
        total_question_length = 0
        total_answer_length = 0
        
        for record in records:
            # Статистика по категориям и разделам
            if 'category' in record:
                stats['categories'][record['category']] += 1
            if 'section' in record:
                stats['sections'][record['section']] += 1
            
            # Средние длины
            if 'question' in record:
                total_question_length += len(record['question'])
            if 'answer' in record:
                total_answer_length += len(record['answer'])
        
        if stats['total'] > 0:
            stats['avg_question_length'] = total_question_length / stats['total']
            stats['avg_answer_length'] = total_answer_length / stats['total']
        
        stats['categories'] = dict(stats['categories'])
        stats['sections'] = dict(stats['sections'])
        
        return stats

