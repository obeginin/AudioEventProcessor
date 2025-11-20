import httpx
import asyncio
import base64
import os
import time
from new_func_hard_2 import AudioEventDetector
import soundfile
import aiofiles
import functools
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from logger_setting import logger_stat


def safe_async(name: str = None):
    """
    Декоратор для безопасного выполнения асинхронных функций:
    - измеряет время выполнения;
    - ловит исключения;
    - пишет лог;
    - возвращает (успех, результат или None).
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            display_name = name or func.__name__
            # Просто берем filename из kwargs если есть
            filename = kwargs.get('filename')
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                if filename:
                    logger_stat.info(f" {display_name} для файла {filename} выполнена за {duration:.2f} сек")
                else:
                    logger_stat.info(f" {display_name} выполнена за {duration:.2f} сек")
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                if filename:
                    logger_stat.error(f" Ошибка в {display_name} для файла {filename}: {e} (время: {duration:.2f} сек)")
                else:
                    logger_stat.error(f" Ошибка в {display_name}: {e} (время: {duration:.2f} сек)")
                logger_stat.debug("Подробности:", exc_info=True)
                return None
        return wrapper
    return decorator


class AudioEventProcessor():
    '''Основной сервис для получения аудио, и дальнейшего анализа'''
    def __init__(self, sm: int = 2, **kwargs):
        self._semaphore = asyncio.Semaphore(sm)                      # Семафор для ограничения количества корутин
        self.process_pool = ThreadPoolExecutor(max_workers=2)        # TODO надо заменить на ProcessPoolExecutor
        self.client = httpx.AsyncClient(timeout=30.0)                # клиент для http запросов
        self.queue = asyncio.Queue()                                 # очередь для записи в лог (крики)
        self.file_lock = asyncio.Lock()                              # Блокировка для файла
        self.writer_task = None                                      # флаг для очереди

        self.batch_size = 10                                         # количество пачек корутн на выполнение
        self.count_tasks = 0                                         # счетчик задач

        self.name_file_list = []                                    # список имен файлов аудиозаписей
        self.dict_audiofiles = {}                                   # словарь для имен аудиофайлов
        self.file_with_names = 'norm.txt'                           # имя файло со списком аудиозаписей

        self.log_file_analys= 'analys_log.json'
        self.directory_scream = 'audio_scream'
        self.directory_interruptions = 'audio_interruptions'

        self.directory = 'temp_audio'                               # каталог для сохранения файлов
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(f'{self.directory}/{self.directory_interruptions}', exist_ok=True)# для файлов с перебиваниями
        os.makedirs(f'{self.directory}/{self.directory_scream}', exist_ok=True)# для файлов с криками

        self.mono_filenames = 'mono_files.txt'                  # файл для сохранения названий моно аудиозаписей
        self.not_files = 'not_files.txt'                        # Файл для сохранения имен для которых отсутствует файл
        self.short_audio = 'short_files.txt'                    # Файл для сохранения имен коротких файлов (меньше 40с)

    async def run_analysis(self):
        '''основная фукнция: читаем из файла все имена, далее находим по имени файл и пачками выполняем обработку файлов'''
        try:
            start_time = time.perf_counter()
            # чтение списка имен из файла

            self.name_file_list = await self.get_filenames_from_file()

            #self.name_file_list = await self.get_filenames_from_folder()
            #print(f"Получен список с имен файлов: {self.name_file_list}")

            self.dict_audiofiles = {}
            all_results = []

            # Разбиваем на пачки
            for i in range(0, len(self.name_file_list), self.batch_size):
                batch = self.name_file_list[i:i + self.batch_size]
                print(f"Обрабатываем пачку {i // self.batch_size + 1}: {batch}")

                # создаем задачи для одновременного запроса данных
                tasks = []
                for filename in batch:
                    self.count_tasks += 1
                    #print(f"{self.count_tasks}: Добавляем задачу для файла: {filename}")
                    tasks.append(asyncio.wait_for(self.process_data(filename=filename), timeout=200.0))
                    self.dict_audiofiles[filename] = self.count_tasks
                batch_results  = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(batch_results)

                print(f"Пачка {i // self.batch_size + 1} завершена")

            #print(f"Словарь из файлов: {len(self.dict_audiofiles)}: {self.dict_audiofiles}")
            #print(f"Количество задач на обработку файлов: {self.count_tasks}")
            # print(f"Задачи: {tasks}")

            # print(f"results={results}")
            # print(f"Получен файл в формает base64: {file_base64}")
            end_time = time.perf_counter() - start_time
            print(f"Время работы программы: {end_time//60} минут")
            # завершаем логирование
            await self.stop()
            return "Закончили"
        except Exception as e:
            print(f"Ичключение в функции func {e}")
            return None

    @safe_async(name='send_request')
    async def send_request(self, *, url, method='GET', headers=None, params=None, data=None, json=None, timeout: float = 50.0):
        """
        Асинхронная функция для отправки HTTP запросов
        Args:
            headers (dict): Заголовки запроса
            params (dict): Query параметры
            data: Form данные
            json: JSON данные
        Returns:
            response: Объект ответа
        """
        try:
            response = await self.client.request(method=method,url=url,headers=headers,params=params,data=data,json=json, timeout=timeout)
            response.raise_for_status()
            return {
                'status': 'success',
                'status_code': response.status_code,
                'data': response.json() if response.content else None,
                'headers': dict(response.headers)
            }
        except httpx.HTTPStatusError as e:
            return {'status': 'error','status_code': e.response.status_code,'error': str(e)}
        except Exception as e:
            return {'status': 'error','error': str(e)}

    async def start_writer(self):
        """Запускает фоновую задачу записи логов"""
        self.writer_scream = asyncio.create_task(self.writer_log_analyze())
        #self.process_pool = ProcessPoolExecutor(max_workers=4)

    async def stop(self):
        """Корректно останавливает запись логов"""
        if self.writer_task:
            await self.queue.put(None)           # Сигнал остановки
            await  self.writer_task

    @safe_async(name='writer_log_analyze')
    async def writer_log_analyze(self):
        '''корутина которая пишет в лог из очереди'''
        async with aiofiles.open(self.log_file_analys, 'a', encoding='utf-8') as f:
            while True:
                item = await self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break

                # Форматируем запись лога
                if isinstance(item, dict):
                    log_line = json.dumps(item, ensure_ascii=False)
                else:
                    log_line = str(item)

                await f.write(f"{log_line}\n")
                # timestamp = datetime.now().isoformat()
                # await f.write(f"{timestamp} - {log_line}\n")
                await f.flush()  # Сразу пишем на диск
                self.queue.task_done()

    async def get_filenames_from_folder(self):
        '''функция возвращает все имена файлов из папки с расширением .mp3'''
        folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\temp_audio"
        # Проверяем существование папки
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует")
            return []
        mp3_list = [f.stem for f in Path(folder_path).glob("*.mp3")]
        print(mp3_list)
        return mp3_list

    async def get_filenames_from_file(self):
        # print(f"Вызван роут get_filenames_from_file")
        # с указанием имени файла
        # self.name_file_list.append('CRMCM-20251010122214362000-id-3236219606')
        return [x.rstrip().split(' ')[-1].split('/')[-1].split('.')[0] for x in open(f'{self.directory}/{self.file_with_names}', 'r')]

    async def get_file_from_AppRecordLoarder(self,*, filename:str, save_flag: bool=True, input_format: str='mp3'):
        response = await self.send_request(url='http://*****/download', params={'filename': filename})
        #print(response)
        '''Получаем файл в формате base64. Если файла нет, фиксируем его имя в общий файл'''
        data = response['data']
        #print(f"data: {data}")
        file_base64 = data.get("recipient_data").get('file')
        #print(f"file_base64: {file_base64}")
        if not file_base64:
            async with self.file_lock:
                async with aiofiles.open(self.not_files, 'a', encoding='utf-8') as f:
                    await f.write(f"{filename}\n")
            #print(f"Нет файла с именем {filename}, записан в лог")
            return None

        # сохраням файл на комп
        if save_flag:
            success = await self.save_file(filename=filename,file_base64=file_base64, input_format=input_format)
            if success is None:
                return None
        return file_base64

    async def open_mp3_to_base64(self,filename):
        async with aiofiles.open(filename, 'rb') as mp3_file:
            mp3_data = await mp3_file.read()
            base64_encoded = base64.b64encode(mp3_data).decode('utf-8')
        return base64_encoded


    async def process_data(self, *, filename: str, input_format: str = 'mp3'):
        async with self._semaphore:
            try:
                t_get_start = time.perf_counter()
                # получаем файл из appRecordLoarder
                file_base64 = await self.get_file_from_AppRecordLoarder(filename=filename)
                if file_base64 is None:
                    return None
                # берем файл из папки windows
                # file_base64 = await self.open_mp3_to_base64(filename=f"{self.directory}/{filename}.mp3")
                t_get = time.perf_counter() - t_get_start
                #id = filename.split('.')[-1].split('-')[-1]

                '''проверяем количество каналов в записи, если 1, то дальше не проверяем, фиксируем в файл, иначе запускаем основные функции'''
                channels = await self.check_audio_channels_async_fast(filename=f"{self.directory}/{filename}.mp3")

                if channels == 1:
                    # TODO пока блокировка, может надо переделать на очередь
                    async with self.file_lock:
                        async with aiofiles.open(self.mono_filenames, 'a', encoding='utf-8') as f:
                            await f.write(f"{filename}\n")
                    #print(f"Моно файл записан в лог: {filename}")
                    await self.delete_file(filename=filename, input_format=input_format)     # удаляем файл
                    return None
                else:
                    # проверяем длительность
                    duration  = await self.check_audio_duration_async_fast(filename=f"{self.directory}/{filename}.mp3")
                    if duration >=40.0:
                        async with self.file_lock:
                            async with aiofiles.open(self.short_audio, 'a', encoding='utf-8') as f:
                                await f.write(f"{filename}\n")
                        await self.delete_file(filename=filename, input_format=input_format)
                        return None

                    start_time = time.perf_counter()
                    analyze = await self.analys_(filename=f"{self.directory}/{filename}.mp3")

                    # ставим в очередь на запись логов
                    end_time = round(time.perf_counter() - start_time, 1)
                    log_entry = {
                        **analyze,
                        "time_get_file": round(t_get,1),
                        #"time_check_channels" : round(t_check,1)
                    }

                    await self.queue.put(log_entry)

                    # сохраняем файлы где были перебивания и крики
                    if analyze.get("overlap_detected") == True:
                        await self.save_file(filename=f"/{self.directory_interruptions}/{filename}", file_base64=file_base64, input_format=input_format)

                    if analyze.get("scream_detected") == True:
                        await self.save_file(filename=f"/{self.directory_scream}/{filename}", file_base64=file_base64, input_format=input_format)
                # после выполнения функции удаляем файл
                await self.delete_file(filename=filename, input_format=input_format)
                return True

            except Exception as e:
                print(f"Ичключение в функции process_data {e} файл: {filename}")
                return None

    @safe_async(name='analys_')
    async def analys_(self, filename: str):
        '''#функция для запуска синхронной функции в потоке'''
        loop = asyncio.get_event_loop()
        # Создаем функцию-обертку для запуска в executor
        def analyze_wrapper():
            detector = AudioEventDetector(config_mode="lite")
            return detector.analyze_audio_complete(filename)

        return await loop.run_in_executor(self.process_pool, analyze_wrapper)

    @safe_async(name='check_audio_channels_async_fast')
    async def check_audio_channels_async_fast(self, *, filename: str):
        """асинхронная проверка каналов"""
        try:
            loop = asyncio.get_event_loop()
            channels = await loop.run_in_executor(None,lambda: soundfile.info(filename).channels)
            return channels
        except Exception as e:
            print(f"Ошибка проверки каналов файла filename: {filename}: {e}")
            return 0

    @safe_async(name='check_audio_duration_async_fast')
    async def check_audio_duration_async_fast(self, *, filename: str):
        """Асинхронная проверка длительности аудио"""
        try:
            loop = asyncio.get_event_loop()
            duration = await loop.run_in_executor(None, lambda: soundfile.info(filename).duration)
            return duration
        except Exception as e:
            print(f"Ошибка проверки длительности файла {filename}: {e}")
            return 0.0  # Возвращаем 0.0 вместо 0

    @safe_async(name='save_file')
    async def save_file(self, *, filename: str, file_base64: str, input_format: str) -> bool:
        """Сохраняет файл на диск"""
        try:
            file_bytes = base64.b64decode(file_base64)  # Декодирование из base64
            if not os.path.exists(f"./{self.directory}"):
                os.makedirs(f"./{self.directory}")
            with open(f"{self.directory}/{filename}.{input_format}", "wb") as file:
                file.write(file_bytes)
            #print(f"Сохранили файл {filename}.{input_format} в каталог {self.directory}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения файла {filename}: {e}")
            raise

    @safe_async(name='delete_file')
    async def delete_file(self,*, filename:str, input_format:str):
        """Удаляет файл по имени из папки"""
        try:
            deleted = False
            file_path = os.path.join(self.directory, f"{filename}.{input_format}")
            if os.path.exists(file_path):
                os.remove(file_path)
                #print(f"Удалили файл {filename}")
                deleted = True
            if not deleted:
                print(f"Файл {filename} не найден в директории {self.directory}")
            return deleted
        except Exception as e:
            print(f"Ошибка удаления файла {filename}: {e}")
            return False




async def main():
    object = AudioEventProcessor()
    await object.start_writer()
    await object.run_analysis()

asyncio.run(main())
