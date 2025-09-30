# Text Summarizer(Для GenAI-1-35)

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
```
## Использование

# Простое использование
```python
from summarizer import summarize

text = "Ваш научный текст здесь..."
summary, word_count = summarize(text)

print("Краткое содержание:")
print(summary)
print(f"Количество слов: {word_count}")
```
## Запуск скрипта
```bash
python summarizer.py
```
Затем введите текст для суммаризации.

## Параметры модели
Модель: facebook/bart-large-cnn

Максимальная длина: 100 слов

Минимальная длина: 30 слов

Усечение: Включено

Сэмплирование: Отключено

## Обработка ошибок
Скрипт включает комплексную обработку ошибок:

Пустой текст: Проверка на пустые или отсутствующие входные данные

Слишком короткий текст: Минимальная длина текста для суммаризации

Ошибки загрузки модели: Проблемы с загрузкой предобученной модели

Ошибки суммаризации: Проблемы в процессе обработки текста моделью

Непредвиденные ошибки: Общий обработчик для непредвиденных ситуаций

## Требования
Python 3.6+

transformers >= 4.0.0

torch

re (входит в стандартную библиотеку Python)

## Ограничения
Предназначен в первую очередь для научных текстов

Максимальная длина входного текста ограничена возможностями модели

Качество суммаризации зависит от исходного текста

## Пример работы
# Вход:
"Недавние исследования в области искусственного интеллекта показали значительный прогресс в обработке естественного языка.
Трансформерные архитектуры, такие как BERT и GPT, revolutionized how machines understand human language.
These advancements have led to improvements in machine translation, text summarization, and question-answering systems."

# Выход:
Краткое содержание:
Recent research in artificial intelligence has shown significant progress in natural language processing.
Transformer architectures like BERT and GPT have revolutionized how machines understand human language, leading to improvements in machine translation and text summarization.

Количество слов: 25

# Примечания
Модель загружается при первом вызове функции summarize, что может занять некоторое время и потребовать значительных ресурсов памяти.






# Text Summarizer with Keyword Annotation(Для GenAI-2-35

Модуль для автоматического создания краткого содержания научных текстов с генерацией аннотаций, включающих заданные ключевые слова.

## Описание

Этот скрипт использует предобученную модель BART-large-CNN от Facebook для генерации сжатых версий научных текстов. Модель автоматически выделяет ключевые моменты исходного текста и создает краткое изложение. Дополнительная функция позволяет гарантировать включение заданных ключевых слов в итоговую аннотацию.

## Особенности

- **Автоматическая суммаризация** научных текстов на английском языке
- **Генерация аннотаций** с обязательным включением ключевых слов
- **Проверка наличия ключевых слов** в результирующем тексте
- **Обработка ошибок** и валидация входных данных
- **Ограничение длины** результата (до 100 слов)

## Установка зависимостей

```bash
pip install transformers torch
```

## Использование
# Основные функции
Функция summarize(text: str) -> Tuple[str, int]
Создает краткое содержание текста.
```python
from text_summarizer import summarize

text = "Your scientific text here..."
summary, word_count = summarize(text)

print("Краткое содержание:")
print(summary)
print(f"Количество слов: {word_count}")
```
Функция generate_annotation_with_keywords(text: str, keywords: List[str]) -> Dict
Генерирует аннотацию с обязательным включением ключевых слов.

```python
from text_summarizer import generate_annotation_with_keywords

text = "Your research paper text..."
keywords = ['neural networks', 'training', 'data']

result = generate_annotation_with_keywords(text, keywords)

print("Аннотация:", result['annotation'])
print("Количество слов:", result['word_count'])
print("Все ключевые слова присутствуют:", result['keywords_present'])
```

