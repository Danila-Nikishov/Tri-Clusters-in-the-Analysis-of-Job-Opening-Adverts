# Вспомогательные функции для парсинга HeadHunter

import os
from unittest import TextTestRunner
import requests

from dotenv import load_dotenv


PROLES_URL = "https://api.hh.ru/professional_roles"
HEADERS = {}

def _ensure_headers() -> None:
    global HEADERS
    if not HEADERS:
        load_dotenv()
        APP_TOKEN = os.getenv("APP_TOKEN")
        APP_NAME = os.getenv("APP_NAME")
        EMAIL = os.getenv("EMAIL")
        HEADERS = {
            'User-Agent': f'{APP_NAME} ({EMAIL})', 
            "Authorization": f"Bearer {APP_TOKEN}"
            }


def parse_proles_handbook(write_to_txt: bool = True) -> list[str]:
    """
    Парсит справочник профессиональных ролей HeadHunter.
    Возвращает профессиональных ролей, отсортированных по ID.
    Каждый элемент списка имеет формат: "  -<id> # <name>"
    Пример: "  -25 # Гейм-дизайнер"
    Опционально: записывает список в .txt файл.
    """
    _ensure_headers()
    proles_handbook = requests.get(PROLES_URL, headers=HEADERS)

    proles_list = []
    for group in proles_handbook.json()["categories"]:
        for prole in group["roles"]:
            id_str = str(prole["id"]) + " " * (3 - len(str(prole["id"])))
            proles_list.append("  -" + id_str + " # " + prole["name"])
    proles_list = list(set(proles_list))
    proles_list.sort(key = lambda x: int(x[3:6]))
    if write_to_txt:
        with open("proles_list.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(proles_list))
    return proles_list