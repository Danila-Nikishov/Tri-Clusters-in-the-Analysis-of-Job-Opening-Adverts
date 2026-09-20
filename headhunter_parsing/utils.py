# Вспомогательные функции для парсинга HeadHunter

import os
import requests
import logging
import json
import time
import random
import pandas as pd

from pandas import DataFrame
from requests import Response
from typing import Any, Optional, Tuple
from tqdm import tqdm
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Parser")


PROLES_URL = "https://api.hh.ru/professional_roles"
VACANCIES_URL = "https://api.hh.ru/vacancies/{}"

HEADERS = {}
N_DAYS_IN_MONTH ={1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
SLEEP_LOWER_BOUNDARY = 0.5
SLEEP_UPPER_BOUNDARY = 1
VACANCY_ATTRIBUTES = [
    "id", "name", "area", "salary", "salary_range", "experience", "description",
    "key_skills", "professional_role", "employer", "work_schedule",
    ]

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


def collect_dates(year: int, month: int) -> list[str]:
    """
    Собирает список дат для парсинга вакансий.
    Возвращает список дат в формате "YYYY-MM-DD".
    """
    n_days = N_DAYS_IN_MONTH.get(month, 31)
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        n_days = 29
    if month < 10:
        month = "0" + str(month)
    dates_list = []
    for day in range(1, n_days + 1):
        if day < 10:
            day = "0" + str(day)
        dates_list.append(f"{year}-{month}-{day}")
    return dates_list


def fetch_ids(
    dates: list[str], relevant_proles: list[str], n_pages_to_parse: int, 
    n_vacancies_per_page: int, vacancies_url: str) -> list[str]:
    """
    Скачивает id вакансий, заданных параметрами.

    dates: список дат, среди которых ищутся вакансии

    relevant_proles: список профессиональных ролей

    n_pages_to_parse: на скольких страницах искать вакансии с фиксированными
                      датой и профессиональной ролью

    n_vacancies_per_page: сколько вакансий выводить на одной поисковой странице
                          Примечание: n_vacancies_per_page <= 100, это внутрен-
                          нее ограничение HeadHunter API

    vacancies_url: базовый адрес, к которому делаются запросы

    headers: заголовки запроса
    """
    _ensure_headers()
    vacancy_ids : set[str] = set()
    for date in tqdm(dates):
        for prole in relevant_proles:
            for page in range(n_pages_to_parse):
                parameters = {
                    'page': str(page),
                    'per_page': str(n_vacancies_per_page),
                    'professional_role': prole,
                    "date_from": date,
                    "date_to": date
                    }

                vacancies = _response_guard(vacancies_url, parameters)
                if not vacancies:
                    continue

                vacancies_json, n_vacancies = _items_extraction_guard(vacancies)
                if not n_vacancies:
                    break

                if (page == n_pages_to_parse - 1) and (n_vacancies == n_vacancies_per_page):
                    logger.warning("Probably, some vacancies are missed")
                logger.info(f"Дата: {date}, профессиональная роль: {prole}, страница: {page}. Найдено вакансий: {n_vacancies}")

                for idx in range(n_vacancies):
                    id = _id_extraction_guard(vacancies_json, idx)
                    if id:
                        vacancy_ids.add(id)

                time.sleep(random.uniform(SLEEP_LOWER_BOUNDARY, SLEEP_UPPER_BOUNDARY))
    return vacancy_ids


def parse_vacancies(path_to_ids: str) -> DataFrame:
    vacancies_df = pd.DataFrame(columns = VACANCY_ATTRIBUTES)
    _ensure_headers()

    good_vacancies = 0
    bad_vacancies = 0

    with open(path_to_ids, 'r', encoding='utf-8') as file:
        idx = 1
        for id in file.readlines():
            id = id.strip()
            new_vacancy = _response_guard(VACANCIES_URL.format(id))
            if not new_vacancy:
                continue
            
            parsed_attributes = _parse_vacancy_guard(new_vacancy, id)
            if parsed_attributes is not None:
                vacancies_df = pd.concat([vacancies_df, parsed_attributes], ignore_index=True) if \
                               not vacancies_df.empty else parsed_attributes
                logger.info(f'Вакансия №{idx} с id: {id}, "{parsed_attributes["name"][0]}" успешно добавлена ✅')
                good_vacancies += 1
            else:
                logger.warning(f'Вакансия №{idx} с id: {id} пропущена ❌')
                bad_vacancies += 1
            idx += 1
            time.sleep(random.uniform(0.5, 1))
        
    vacancies_df.to_csv('data/vacancies_raw.csv', index=False)
    print(f'''
    Успешно добавлено {good_vacancies} вакансий
    При добавлении {bad_vacancies} вакансий возникли ошибки
    '''
    )
    return vacancies_df


def _parse_vacancy_guard(vacancy: Response, id: str) -> DataFrame:
    try:
        vacancy_json = vacancy.json()
    except Exception as e: 
        logger.warning(f"Something wrong with converting Response to json. Vacancy id: {id}\n{e}")
        return
    attributes = []
    try:
        attributes.append(vacancy_json.get("id", ""))
        attributes.append(vacancy_json.get("name", ""))
        attributes.append((vacancy_json.get("area", "")).get("name", ""))
        attributes.append(vacancy_json.get("salary", ""))
        attributes.append(vacancy_json.get("salary_range", ""))
        attributes.append((vacancy_json.get("experience", "")).get("name", ""))
        attributes.append(vacancy_json.get("description", ""))
        attributes.append(vacancy_json.get("key_skills", ""))
        attributes.append((vacancy_json.get("professional_roles", [{}]))[0].get("id", ""))
        attributes.append((vacancy_json.get("employer", "")).get("name", ""))
        attributes.append((vacancy_json.get("work_schedule_by_days", [{}]))[0].get("name", ""))
    except Exception as e:
        logger.warning(f"Something wrong with extracting attributes. Vacancy id: {id}\n{e}")
        return
    try:
        new_row = pd.DataFrame(data = [attributes], columns = VACANCY_ATTRIBUTES)
        return new_row
    except Exception as e:
        logger.warning(f"Something wrong with converting attributes to DataFrame. Vacancy id: {id}\n{e}")
        return
        
    


# ---------------------------------------------------------------------------------------------------------------------
# Защиты от ошибок
# ---------------------------------------------------------------------------------------------------------------------


def _response_guard(url: str, params: dict[str, Any] = None) -> Optional[Response]:
    try:
        response = requests.get(url = url, params = params, headers = HEADERS)
        return response
    except Exception as e: 
        logger.warning(f"Something wrong with request with the parameters: {params}\n{e}")


def _items_extraction_guard(vacancies: Response) -> Optional[Tuple[dict, int]]:
    try:
        vacancies_json = vacancies.json()
    except Exception as e: 
        logger.warning(f"Something wrong with converting Response to json: {vacancies}\n{e}")
        return (None, None)
    try:
        n_vacancies = len(vacancies_json.get('items'))
        return (vacancies_json, n_vacancies)
    except:
        logger.warning(f"Something wrong with extraction items from json: {vacancies}\n{e}")
        return (None, None)


def _id_extraction_guard(vacancies_json: dict[str, Any], idx: int) -> Optional[str]:
    try:
        id = vacancies_json['items'][idx]['id']
        return id
    except Exception as e:
        logger.warning(f"Vacancy id wasn't added: idx={idx}\n{e}")