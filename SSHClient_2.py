import asyncio
import paramiko
from typing import List, Optional
import logging
import subprocess
import aiofiles
import time

'''Версия 2.0 дописал общую функцию, которая по очереди вызывает два метода класса, теперь её можно импортировать в другое приложение
так же добавляю метод, который ищет дубликаты записей в файле'''
class AsyncSSHClient:
    def __init__(self):
        """
        Асинхронный SSH клиент для выполнения команд на удаленном сервере
        Args:
            base_search_path: Базовый путь для поиска
            connect_timeout: Таймаут подключения
        """

        self.host = ""
        self.username = "b"
        self.password = ""
        self.port = 22
        self.base_search_path = "/store/"         # каталог в котором ищем папки
        self.maxdepth = 1                                   # уровень вложенности папок
        self.date_path = "2025/10/10"                       # дата по которой ищем
        self.exclude_folder = ""               # исключаем папку
        self.connect_timeout = 10
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(5)
        self.file_lock = asyncio.Lock()
        self.files_in_archives = 'audio_in_archives.txt'    # файл с сохранение имен всех файлов из архивов
        self.files_in_folders = 'audio_in_folders.txt'  # файл с сохранение имен всех файлов из архивов
        self.tar_list = []                                  # список с именами всех архивов
        self.audio_in_tar = 0                               # счетчик количества файлов во архивах
        self.count_audio = 0
        self.count_all_audio = 0 # файлы без архивов

    async def connect(self) -> None:
        """Асинхронное подключение к SSH серверу"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._connect_sync)
            self.logger.info(f"Успешное подключение к {self.host}")
        except Exception as e:
            self.logger.error(f"Ошибка подключения к {self.host}: {e}")
            raise

    def _connect_sync(self) -> None:
        """Синхронное подключение (выполняется в executor)"""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=self.host,username=self.username,password=self.password,port=self.port,timeout=self.connect_timeout)

    async def close(self) -> None:
        """Закрытие SSH соединения"""
        if self.ssh_client:
            await asyncio.get_event_loop().run_in_executor(None, self.ssh_client.close)
            self.ssh_client = None
            self.logger.info(f"Соединение с {self.host} закрыто")


    async def execute_command(self, command: str) -> List[str]:
        """Асинхронное выполнение команды и возврат результатов
        Args: command: Команда для выполнения
        Returns: List[str]: Список строк с результатами
        """
        if not self.ssh_client: await self.connect()

        try:
            # Выполнение команды в executor
            stdin, stdout, stderr = await asyncio.get_event_loop().run_in_executor(None, self.ssh_client.exec_command, command)

            # Чтение результатов
            output = await asyncio.get_event_loop().run_in_executor(None, stdout.read)
            errors = await asyncio.get_event_loop().run_in_executor(None, stderr.read)

            output_text = output.decode('utf-8').strip()
            error_text = errors.decode('utf-8').strip()

            if error_text:
                self.logger.warning(f"Stderr при выполнении команды: {error_text}")

            # Разделяем результат на строки и фильтруем пустые
            results = [line for line in output_text.split('\n') if line]
            return results

        except Exception as e:
            self.logger.error(f"Ошибка выполнения команды '{command}': {e}")
            raise

    async def execute_command_streaming(self, command: str):
        """Выполнение команды с потоковым выводом (результаты по мере появления)"""
        if not self.ssh_client:
            await self.connect()

        try:
            stdin, stdout, stderr = await asyncio.get_event_loop().run_in_executor(None, self.ssh_client.exec_command, command)

            # Читаем построчно в реальном времени
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, stdout.readline)
                if not line:
                    break
                yield line.decode('utf-8').strip()

        except Exception as e:
            self.logger.error(f"Ошибка выполнения команды: {e}")
            raise

    async def find_folders(self) -> List[str]:
        """ Поиск папок в базовом пути
        Args: maxdepth: Глубина поиска
        Returns: List[str]: Список найденных папок
        """
        find_folders_command = f'find "{self.base_search_path}" -maxdepth {self.maxdepth} -type d ! -name "{self.exclude_folder}" -type d 2>/dev/null'
        self.logger.info(f"Выполнение поиска: {find_folders_command}")

        return await self.execute_command(find_folders_command)

    async def find_tar_archives(self, search_path: str) -> List[str]:
        """Поиск всех tar архивов по указанному пути и подпапкам"""
        async with self._semaphore:
            try:
                check_cmd = f'[ -d "{search_path}" ] && echo "exists" || echo "not_exists"'
                check_result = await self.execute_command(check_cmd)

                if not check_result or check_result[0] != "exists":
                    self.logger.info(f"Директория {search_path} не существует, пропускаем")
                    return None
                command = f'find "{search_path}" -name "*.tar" -type f'
                tar_files = await self.execute_command(command)

                if len(tar_files) != 0:
                    self.logger.info(f"Найдено tar архивов в {"/".join(search_path.split("/")[:4])}: {len(tar_files)}")
                    self.tar_list.extend(tar_files)

                return True

            except Exception as e:
                self.logger.error(f"Ошибка при поиске архивов: {e}")
                return None

    async def process_archive_for_audio(self, archive_path): #, output_file):
        """
        Ищет MP3 файлы в архиве .tar
        """
        async with self._semaphore:
            try:
                # Команда для списка MP3 файлов в tar архиве
                list_command = f'tar -tvf "{archive_path}" | grep "\.mp3$"'

                lines = await self.execute_command(list_command)
                filename_list = []

                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            size_str = parts[2]  # размер файла
                            filename = parts[5]  # имя файла

                            try:
                                size_bytes = int(size_str)
                                # Проверяем условия: имя заканчивается на .mp3 и размер >200KB
                                if filename.lower().endswith('.mp3'): # and size_bytes > 204800 :
                                    #output_file.write(f"{filename}\n")
                                    filename_list.append(filename)
                                    #self.logger.info(f"файл mp3 в архиве: {filename}")
                            except ValueError:
                                continue

                # записываем в файл
                self.audio_in_tar += len(filename_list)
                if len(filename_list)!=0: await self.save_results(output_file=self.files_in_archives, file_list=filename_list, archive_name=archive_path)
                return len(filename_list)

            except Exception as e:
                print(f"Ошибка обработки архива {archive_path}: {e}")
                return 0

    async def save_results(self, output_file: str, file_list: list, archive_name: str):
        """Сохраняет все найденные имена файлов, каждое с новой строки"""
        async with self.file_lock: # блоикруем файл (хотя может зря
            async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                content = '\n'.join(file_list)
                await f.write(content)
                self.logger.info(f"✅ Сохранено {len(file_list)} файлов {archive_name}")
                self.count_all_audio += len(file_list)

            return len(file_list)


    async def search_mp3_files_in_folders(self, min_size_kb: int = 200) -> dict:
        """Оптимизированный поиск - одна SSH команда для всех папок"""
        async with self._semaphore:
            try:
                folders = await self.find_folders()
                #self.logger.info(f"📁 Найдено папок: {len(folders)}")

                if not folders:
                    return {'success': True, 'files_found': 0}

                # Создаем пути для поиска
                search_paths = [f"{folder}/{self.date_path}" for folder in folders]

                # ОДНА команда для поиска во всех папках
                paths_string = " ".join(f'"{path}"' for path in search_paths)
                search_command = f'find {paths_string} -name "*.mp3" -type f 2>/dev/null'
                self.logger.info(f"🔍 Выполняем поиск во всех папках одной командой: {paths_string}")
                file_paths = await self.execute_command(search_command)

                # Обрабатываем результаты
                filenames = []
                for file_path in file_paths:
                    if file_path.strip():
                        filename = file_path.split('.')[-2].split('/')[-1]
                        filenames.append(filename)

                if filenames:
                    await self.save_results(output_file=self.files_in_folders, file_list=filenames, archive_name='из всех папок')
                    self.count_audio = len(filenames)
                    #self.logger.info(f"✅ Найдено {len(filenames)} MP3 файлов")

                return {
                    'success': True,
                    'files_found': len(filenames),
                    'folders_searched': len(folders)
                }

            except Exception as e:
                self.logger.error(f"❌ Ошибка: {e}")
                return {'success': False, 'error': str(e)}

    async def search_mp3_service(self) -> dict:
        """Функция для поиска MP3 файлов в папках (без архивов)"""
        try:
            start_time = time.perf_counter()
            result = await self.search_mp3_files_in_folders()
            result['execution_time_seconds'] = round(time.perf_counter() - start_time, 1)
            logging.info(f"Общее количество найденных аудио файлов в папках:{self.count_audio}")
            logging.info(f"Время поиска аудио файлов в папках: {result['execution_time_seconds']}c")
            return result
        except Exception as e:
            logging.error(f"❌ Ошибка при поиске mp3 файлов: {e}")
            return {
                'success': False,
                'error': str(e),
            }

    async def search_mp3_in_archive(self) -> dict:
        """
        Основная функция для mp3 файлов внутри архивов
        Returns: dict: Результаты поиска"""
        try:
            start_time = time.perf_counter()

            # Поиск папок
            folders = await self.find_folders()
            logging.info(f"📁 Найдено папок: {len(folders)}")

            # Поиск архивов в папках
            tasks_folders = []
            for folder in folders:
                path = f"{folder}/{self.date_path}"
                # logging.info(f"{path}")
                task = asyncio.create_task(self.find_tar_archives(search_path=path))
                tasks_folders.append(task)

            await asyncio.gather(*tasks_folders)
            logging.info(f"📦 Найдено архивов: {len(self.tar_list)}")

            # Обработка архивов
            tasks_tar = []
            for tar in self.tar_list:
                task = asyncio.create_task(self.process_archive_for_audio(tar))
                tasks_tar.append(task)

            await asyncio.gather(*tasks_tar)

            logging.info(f"Общее количество найденных архивов:{len(self.tar_list)}")
            logging.info(f"Список найденных архивов:{self.tar_list}")
            logging.info(f"Всего файлов сохранено из архивов: {self.audio_in_tar}")
            end_time = round(time.perf_counter() - start_time, 1)
            logging.info(f"Время поиска аудио файлов в архивах:: {end_time}c")

            return {
                'success': True,
                'execution_time_seconds': end_time,
                'folders_searched': len(folders),  # Количество папок для поиска
                'total_archives': len(self.tar_list),  # Количество архивов
                'mp3_files_in_archive': self.audio_in_tar,  # Количество фалов в архивах
                'date_searched': self.date_path,
                # 'archives_sample': ssh_client.tar_list[:5],  # первые 5 для примера
            }

        except Exception as e:
            logging.error(f"❌ Ошибка при поиске архивов: {e}")
            return {
                'success': False,
                'error': str(e),
            }


    async def request_appSimChecker(self) -> dict:
        """Функция для поиска MP3 файлов в папках (без архивов)"""
        try:
            command = '''cat /home/arhipov-sm/AppSimChecker/results/2025-10-23.json'''
            result = await self.execute_command(command)
            logging.info(f"Результат: {result}")
            return result
        except Exception as e:
            logging.error(f"❌ Ошибка при поиске mp3 файлов: {e}")
            return {
                'success': False,
                'error': str(e),
            }




async def search_all_audio_service(mode: str = 'all'):
    """Общая функция для поиска MP3 файлов в папках и архивах
    Args:
        mode: Режим поиска
            - 'folder': только поиск в папках
            - 'archive': только поиск в архивах
            - 'all': поиск и в папках и в архивах
    """
    ssh_client = AsyncSSHClient()
    try:
        start_time = time.perf_counter()
        await ssh_client.connect()
        logging.info("✅ SSH подключение установлено")
        # поиск по папкам
        if mode in ['folder', 'all']:
            folder_results = await ssh_client.search_mp3_service()
            #print(folder_results)

        # поиск по архивам
        if mode in ['archive', 'all']:
            archive_results = await ssh_client.search_mp3_in_archive()

            #time_archive = time.perf_counter() - time_folder
            #print(archive_results)
        total_time = round(time.perf_counter() - start_time, 1)
        logging.info(f"Общее количество найденных файлов: {ssh_client.count_all_audio}")
        logging.info(f"total_time_seconds: {total_time}c")
        # return {
        #     'success': True,
        #     'total_time_seconds': total_time,
        #     'time_folder': time_folder,
        #     'time_archive': time_archive,
        #     'folder_search': folder_results,
        #     'archive_search': archive_results
        # }

    except Exception as e:
        logging.error(f"❌ Ошибка при поиске аудио: {e}")
        return {
            'success': False,
            'error': str(e),
        }
    finally:
        await ssh_client.close()


# Пример использования
async def main():
    logging.basicConfig(level=logging.INFO)
    await search_all_audio_service(mode='all') #  'folder' 'archive' 'all'


if __name__ == "__main__":
    asyncio.run(main())