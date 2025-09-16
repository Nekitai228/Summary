from transformers import pipeline
import re
from typing import Tuple, Optional
def summarize(text: str) -> Tuple[str,int]:
    """
    Создаёт краткое содержание научного текста (максимум 100 слов)
    Возвращает пару из сжатого текста и количества его слов

    Args:
        text(str): Входной текст для суммаризации

    Returns:
        Tuple[str,int]: Кортеж из краткого содержания и количества слов

    Raises:
        ValueError: Если текст пустой, слишком короткий или содержит только пробелы
        RuntimeError: Если произошла ошибка при загрузке модели или суммаризации
        Exception: Для других непредвиденных ошибок
    """

    try:
        if not text or not isinstance(text,str):
            raise ValueError("Текст не может быть пустым или не строкового типа")
        #Инициализация модели
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as model_error:
            raise RuntimeError(f"Ошибка загрузки модели: {model_error}")
    
        #Очистка текста от лишних пробелов
        text_clean = re.sub(r'\s+', ' ', text).strip()
        if not text_clean:
            raise ValueError("Текст содержит только пробелы или пуст")
        if len(text_clean.split()) < 10:
            raise ValueError("Текст слишком короткий для суммаризации(минимум 10 слов)")
    
        #Суммаризация
        try:
            result = summarizer(
                text_clean,
                max_length=100,
                min_length=30,
                do_sample=False,
                truncation=True
            )
        except Exception as summarization_error:
            raise RuntimeError(f"Ошибка во время суммаризации: {summarization_error}")

    
        #Получение и обрезка summary
        summary = result[0]['summary_text']
        words = summary.split()[:100]
        final_summary = ' '.join(words)

        return final_summary, len(words)

    except (ValueError, RuntimeError):
        # Перевыбрасываем ожидаемые исключения
        raise
    except Exception as unexpected_error:
        # Обрабатываем непредвиденные ошибки
        raise Exception(f"Непредвиденная ошибка: {unexpected_error}")

#Использование 
def main():
    """Основная функция для запуска скрипта"""
    try:
        prompt = input("Введите текст для суммаризации: ")
        
        if not prompt.strip():
            print("Ошибка: Введен пустой текст")
            return
            
        summary, count_words = summarize(prompt)
        print("\nКраткое содержание:")
        print(summary)
        print(f"\nКоличество слов: {count_words}")
        
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

