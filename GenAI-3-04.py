from transformers import pipeline
import re
from typing import Tuple, Dict, List
import os
from keybert import KeyBERT
import datetime

# Глобальные константы для параметров
MODEL_NAME = "facebook/bart-large-cnn"
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 50
MIN_INPUT_WORDS = 10

# Глобальные переменные для моделей
summarizer = None
kw_model = None

def initialize_models():
    """Инициализация моделей один раз при запуске"""
    global summarizer, kw_model
    
    try:
        if summarizer is None:
            summarizer = pipeline("summarization", model=MODEL_NAME)
            print(f"Модель для суммаризации {MODEL_NAME} успешно загружена")
        
        if kw_model is None:
            kw_model = KeyBERT()
            print("Модель для извлечения ключевых слов успешно загружена")
            
    except Exception as model_error:
        raise RuntimeError(f"Ошибка загрузки моделей: {model_error}")

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Извлекает ключевые слова из текста
    
    Args:
        text (str): Исходный текст
        num_keywords (int): Количество извлекаемых ключевых слов
        
    Returns:
        List[str]: Список ключевых слов
    """
    try:
        # Извлекаем ключевые слова
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=num_keywords,
            diversity=0.5
        )
        
        # Возвращаем только ключевые слова (без оценок)
        return [kw[0] for kw in keywords]
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при извлечении ключевых слов: {e}")

def summarize(text: str) -> Tuple[str, int]:
    """
    Создаёт краткое содержание научного текста (максимум 150 слов)
    
    Args:
        text (str): Входной текст для суммаризации

    Returns:
        Tuple[str, int]: Кортеж из краткого содержания и количества слов
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Текст не может быть пустым или не строкового типа")
        
        # Очистка текста от лишних пробелов
        text_clean = re.sub(r'\s+', ' ', text).strip()
        if not text_clean:
            raise ValueError("Текст содержит только пробелы или пуст")
        
        if len(text_clean.split()) < MIN_INPUT_WORDS:
            raise ValueError(f"Текст слишком короткий для суммаризации (минимум {MIN_INPUT_WORDS} слов)")

        # Суммаризация
        result = summarizer(
            text_clean,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False,
            truncation=True
        )

        # Получение и обрезка summary
        summary = result[0]['summary_text']
        words = summary.split()[:MAX_SUMMARY_LENGTH]
        final_summary = ' '.join(words)

        return final_summary, len(words)

    except Exception as e:
        raise RuntimeError(f"Ошибка при суммаризации: {e}")

def generate_annotation_with_keywords(text: str, keywords: List[str]) -> Dict:
    """
    Генерирует аннотацию с обязательным включением заданных ключевых слов
    
    Args:
        text (str): Исходный текст статьи
        keywords (List[str]): Список ключевых слов для включения
        
    Returns:
        Dict: Словарь с результатами генерации аннотации
    """
    try:
        # Суммаризируем исходный текст
        summary, word_count = summarize(text)
        
        # Проверяем наличие ключевых слов в аннотации
        missing_keywords = [kw for kw in keywords if kw.lower() not in summary.lower()]
        
        # Если каких-то ключевых слов нет, добавляем их
        if missing_keywords:
            # Создаем дополнение с недостающими ключевыми словами
            addition = f" The research focuses on {', '.join(missing_keywords)}."
            new_summary = summary + addition
            
            # Обновляем счетчик слов и обрезаем если превышен лимит
            words = new_summary.split()
            if len(words) > MAX_SUMMARY_LENGTH:
                new_summary = ' '.join(words[:MAX_SUMMARY_LENGTH])
            
            summary = new_summary
            word_count = len(summary.split())
        
        return {
            'annotation': summary,
            'word_count': word_count,
            'keywords': keywords,
            'keywords_present': all(kw.lower() in summary.lower() for kw in keywords),
            'missing_keywords': missing_keywords
        }
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при генерации аннотации: {e}")

def save_results(annotation_data: Dict, original_text: str, filename: str = None):
    """
    Сохраняет результаты в файл
    
    Args:
        annotation_data (Dict): Данные аннотации
        original_text (str): Исходный текст
        filename (str): Имя файла для сохранения
    """
    try:
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotation_results_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("РЕЗУЛЬТАТЫ АННОТАЦИИ СТАТЬИ\n")
            f.write("="*60 + "\n\n")
            
            f.write("ИСХОДНЫЙ ТЕКСТ:\n")
            f.write("-" * 40 + "\n")
            f.write(original_text[:500] + "..." if len(original_text) > 500 else original_text)
            f.write("\n\n")
            
            f.write("АННОТАЦИЯ:\n")
            f.write("-" * 40 + "\n")
            f.write(annotation_data['annotation'])
            f.write(f"\n\nКоличество слов: {annotation_data['word_count']}\n\n")
            
            f.write("КЛЮЧЕВЫЕ СЛОВА:\n")
            f.write("-" * 40 + "\n")
            for keyword in annotation_data['keywords']:
                status = "✓" if keyword.lower() in annotation_data['annotation'].lower() else "✗"
                f.write(f"{status} {keyword}\n")
            
            f.write(f"\nВсе ключевые слова присутствуют: {'Да' if annotation_data['keywords_present'] else 'Нет'}\n")
            
            if annotation_data['missing_keywords']:
                f.write(f"Добавленные ключевые слова: {', '.join(annotation_data['missing_keywords'])}\n")
            
            f.write(f"\nФайл сохранен: {filename}\n")
            f.write(f"Время создания: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Результаты сохранены в файл: {filename}")
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении файла: {e}")

def main():
    """Основная функция для запуска скрипта"""
    try:
        # Инициализируем модели
        initialize_models()
        
        print("="*50)
        print("СИСТЕМА АВТОМАТИЧЕСКОЙ АННОТАЦИИ СТАТЕЙ")
        print("="*50)
        
        # 1. Получение текста статьи
        print("\n1. ВВЕДИТЕ ТЕКСТ СТАТЬИ:")
        print("(Для завершения ввода введите пустую строку)")
        
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        
        original_text = " ".join(lines)
        
        if not original_text.strip():
            print("Ошибка: Введен пустой текст")
            return
        
        # 2. Извлечение ключевых слов
        print("\n2. ИЗВЛЕЧЕНИЕ КЛЮЧЕВЫХ СЛОВ...")
        keywords = extract_keywords(original_text, num_keywords=5)
        print(f"Извлеченные ключевые слова: {', '.join(keywords)}")
        
        # 3. Генерация аннотации
        print("\n3. ГЕНЕРАЦИЯ АННОТАЦИИ...")
        result = generate_annotation_with_keywords(original_text, keywords)
        
        # Вывод результатов
        print("\n" + "="*50)
        print("АННОТАЦИЯ:")
        print("="*50)
        print(result['annotation'])
        print(f"\nКоличество слов: {result['word_count']}")
        
        print("\nКЛЮЧЕВЫЕ СЛОВА:")
        for keyword in result['keywords']:
            status = "✓" if keyword.lower() in result['annotation'].lower() else "✗"
            print(f"  {status} {keyword}")
        
        # 4. Сохранение результатов
        print("\n4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        save_results(result, original_text)
        
        print("\nПроцесс завершен успешно!")
        
    except ValueError as ve:
        print(f"Ошибка входных данных: {ve}")
    except RuntimeError as re:
        print(f"Ошибка обработки: {re}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")

if __name__ == "__main__":
    main()