# Вспомогательные функции для извлечения и обработки аспектов вакансий


import re
import pymorphy3

from functools import lru_cache


_MORPH_INSTANCE = None
_CACHE_SIZE = 200000


def _ensure_morph_instance() -> None:
    global _MORPH_INSTANCE
    if _MORPH_INSTANCE is None:
        _MORPH_INSTANCE = pymorphy3.MorphAnalyzer()


def text_cleaning(text: str, PLUSES: list[str], DOTS: list[str], MINUSES: list[str], 
                  RUBBISH_TAGS: list[str], RUBBISH_PUNCTUATION: list[str], TAGS_TO_SAVE: list[str],
                  PUNCTUATION_TO_DELETE_ON_THE_BOUNDARIES: list[str]) -> str:
    """
    Функция очистки и нормализации описания вакансии.
    Описание значения аргументов смотреть в token_extraction.yaml.
    Что происходит, по шагам: 
    1) Приведение к нижнему регистру
    2) Замена буквы "ё" на букву "е"
    3) Работа с пунктуацией:
    - Замена символов, похожих на плюс, на +
    - Замена символов, похожих на точку, на .
    - Замена символов, похожих на минус, на -
    - Удаление бессодержательной пунктуации
    - Удаление содержательной пунктуации в конце слов
    4) Удаление бессодержательных тегов, сохранение содержательных
    5) Удаление лишних пробелов
    """

    cleaned_text = text.lower()                                                        # Шаг 1

    cleaned_text = cleaned_text.replace('ё', 'е')                                      # Шаг 2

    for plus in PLUSES:                                                                # Шаг 3
        cleaned_text = cleaned_text.replace(plus, '+')
    for dot in DOTS:
        cleaned_text = cleaned_text.replace(dot, '.')
    for minus in MINUSES:
        cleaned_text = cleaned_text.replace(minus, '-')
    for rubbish_char in RUBBISH_PUNCTUATION:
        cleaned_text = cleaned_text.replace(rubbish_char, ' ')
    for char in PUNCTUATION_TO_DELETE_ON_THE_BOUNDARIES:
        escaped_char = re.escape(char)
        cleaned_text = re.sub(rf'([\w]+){escaped_char}(?!\w)', r'\1 ', cleaned_text)
        cleaned_text = re.sub(rf'(?<!\w){escaped_char}(\w+)', r' \1', cleaned_text)
        cleaned_text = re.sub(rf'(?:\s|^){escaped_char}(?:\s|$)', r' ', cleaned_text)

    for tag in RUBBISH_TAGS:                                                           # Шаг 4
        cleaned_text = cleaned_text.replace(tag, ' ')
    for tag in TAGS_TO_SAVE:
        cleaned_text = cleaned_text.replace(tag, f' {tag} ')

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)                                   # Шаг 5
    cleaned_text = cleaned_text.strip()

    return cleaned_text


@lru_cache(maxsize = _CACHE_SIZE)
def _normalize_word(word: str) -> str:
    _ensure_morph_instance()
    try:
        normalized = _MORPH_INSTANCE.parse(word)[0].normal_form
        return normalized
    except:
        return word 


def normalize_sentence(sentence: str) -> str:
    words = sentence.strip().split()
    return ' '.join(_normalize_word(word) for word in words if word)


def classify_sentence(
    sentence: str, REQUIREMENTS_KEY_WORDS: list[str],
    RESPONSIBILITIES_KEY_WORDS: list[str], CONDITIONS_KEY_WORDS: list[str]
    ) -> (bool, list[bool]):
    sent_types = [0, 0, 0]
    words = set(sentence.split(' '))
    for key in REQUIREMENTS_KEY_WORDS:
        if key in words:
            sent_types[0] = 1
            break
    for key in RESPONSIBILITIES_KEY_WORDS:
        if key in words:
            sent_types[1] = 1
            break
    for key in CONDITIONS_KEY_WORDS:
        if key in words:
            sent_types[2] = 1
            break
    is_change = int(sum(sent_types) > 0)
    return is_change, sent_types