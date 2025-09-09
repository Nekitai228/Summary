from transformers import pipeline
import re
def summarize(text: str) -> tuple:
    #Создаёт краткое содержание научного текста(максимум 100 слов)
    #Возвращает пару из сжатого текста и количества его слов


    #Инициализация модели
    summarizer = pipeline("summarization", model = "facebook/bart-large-cnn") #обученная модель для создания кратких содержаний

    #Очистка текста от лишних пробелов
    text_clean = re.sub(r'\s+', ' ', text).strip()

    #Суммаризация
    result = summarizer(
        text_clean,
        max_length = 100,
        min_length = 30,
        do_sample = False,
        truncation = True
    )

    #Получение и обрезка summary
    summary = result[0]['summary_text']
    words = summary.split()[:100]
    final_summary = ' '.join(words)

    return final_summary, len(words)


#Использование 
promt = input()

summary, count_words = summarize(promt)
print("Краткое содержание:")
print(summary)
print(f"\nКоличество слов: {count_words}")