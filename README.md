# Text Summarizer

Модуль для автоматического создания кратких содержаний научных текстов с использованием модели машинного обучения.

## Описание

Данный скрипт использует предобученную модель BART-large-CNN от Facebook для генерации сжатых версий научных текстов. Модель автоматически выделяет ключевые идеи и основные положения исходного текста, создавая краткое содержание длиной до 100 слов.

## Особенности

- Автоматическое суммирование научных текстов
- Ограничение результата до 100 слов
- Очистка текста от лишних пробелов и форматирования
- Возвращает как сжатый текст, так и количество слов в нем

## Установка зависимостей

```bash
pip install transformers torch

Использование

Простое использование
from summarizer import summarize

text = "Ваш научный текст здесь..."
summary, word_count = summarize(text)

print("Краткое содержание:")
print(summary)
print(f"Количество слов: {word_count}")

Запуск скрипта
python summarizer.py

Затем введите текст для суммаризации.

Параметры модели
Модель: facebook/bart-large-cnn

Максимальная длина: 100 слов

Минимальная длина: 30 слов

Усечение: Включено

Сэмплирование: Отключено

Требования
Python 3.6+

transformers >= 4.0.0

torch

re (входит в стандартную библиотеку Python)

Ограничения
Предназначен в первую очередь для научных текстов

Максимальная длина входного текста ограничена возможностями модели

Качество суммаризации зависит от исходного текста

Пример работы
Вход:
"Недавние исследования в области искусственного интеллекта показали значительный прогресс в обработке естественного языка.
Трансформерные архитектуры, такие как BERT и GPT, revolutionized how machines understand human language.
These advancements have led to improvements in machine translation, text summarization, and question-answering systems."
Выход:
Краткое содержание:
Recent research in artificial intelligence has shown significant progress in natural language processing.
Transformer architectures like BERT and GPT have revolutionized how machines understand human language, leading to improvements in machine translation and text summarization.

Количество слов: 25

Примечания
Модель загружается при первом вызове функции summarize, что может занять некоторое время и потребовать значительных ресурсов памяти.
