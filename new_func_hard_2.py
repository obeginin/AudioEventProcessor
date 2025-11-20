
from dataclasses import dataclass, field
import numpy as np
import librosa
import scipy.signal as sig
import matplotlib.pyplot as plt
import glob
import os
import json
import time
from datetime import datetime
from pathlib import Path

np.seterr(invalid='ignore', divide='ignore')


@dataclass
class AudioEventConfig:
    # базовые параметры конфигурации (примеры и описание) берутся из конфига json
    sr: int = 22050                                 # Количество сэмплов (отсчетов) аудио в секунду
    scream_window_sec: float = 2.0                  # длительность окна анализа крика
    scream_hop_sec: float = 0.5                     # шаг между окнами
    frame_length: int = 2048                        # размер окна для спектра
    frame_hop: int = 512                            # шаг между соседними окнами
    blank_pct: float = 0.3                          # доля самых тихих фреймов-фон (от 0 до 1)
    rms_multiplier: float = 2.5                     # во сколько раз громкость должна превышать фоновый шум
    flux_multiplier: float = 3.5                    # во сколько раз спекральные изменения должны превышать фон
    min_scream_sec: float = 0.8                     # минимальная длительнось крика
    max_silence_between_screams: float = 0.1        # максимальная пауза между криками (для объединения)
    high_freq_low: int = 3000                       # нижняя граница диапазона для определения крика
    high_freq_high: int = 8000                      # верхняя граница диапазона для определения крика
    high_freq_threshold: float = 0.04               # Минимальная энергия в высокочастотном диапазоне
    silence_threshold_db = -45                      # порог тишины для крика
    # параметры перебиваний
    overlap_chunk_sec = 0.1,                        # размер окна (чанка) в секундах,
    min_overlap_sec = 1.5,                          # минимальный интервал перебиваний ( в секундах)
    rms_threshold_overlap = -34                     # порог громкости для перебиваний (в dB)



class AudioEventDetector:
    '''класс для определения крика и перебиваний в аудиозаписи'''
    def __init__(self, config_mode="lite", config_path="config_json.json"):
        config = self._load_json(config_path)       # 1. Загружаем JSON
        if config_mode not in config:               # 2. Берём нужный профиль
            raise ValueError(f"Profile '{config_mode}' not found in {config_path}")
        profile = config[config_mode]

        # 3. Разворачиваем все параметры прямо в self
        for key, value in profile.items():
            setattr(self, key, value)
        self.config_mode = config_mode

        self.sr = 22050
        self.frame_length = 2048
        self.frame_hop = 512

    def _load_json(self, config_path):
        """Приватный метод загрузки JSON"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file '{path}' not found")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------ вспомогательные методы ------------------
    def bandpass(self, y: np.ndarray, low: float = 500, high: float = 4000, order: int = 6):
        ''' Фильтр: Усилить сигнал в речевом диапазоне и подавить низкочастотные (гул, шум ветра) и высокочастотные (шипение, некоторые электронные звуки) помехи'''
        ny = self.sr / 2
        b, a = sig.butter(order, [low/ny, high/ny], btype='band')
        return sig.filtfilt(b, a, y)
    def compute_rms(self, y):
        '''Вычисление RMS громкости'''
        return librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.frame_hop)[0]

    def compute_flux(self, y):
        '''Спектральный поток (изменчивость спектра)
        результатом явялется массив, который показывает как сильно меняется спектр во времени (высокие значения = резкие изменения)'''
        S = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.frame_hop)) # STFT спектрограмма
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))                           # Изменения между соседними фреймами
        return flux / (np.max(flux) + 1e-8)

    def high_freq_energy(self, chunk):
        '''Энергия высоких частот (Крики содержат много высокочастотной энергии)'''
        ny = self.sr / 2
        b, a = sig.butter(4, [self.high_freq_low/ny, self.high_freq_high/ny], btype='band')
        y_high = sig.filtfilt(b, a, chunk)
        return np.sqrt(np.mean(y_high**2))

    def load_audio(self, audio_path):
        """Загружает аудиофайл с помощью librosa"""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=False)
        return y, sr

    def process_channel(self, channel_y):
        """Предобработка канала. Применяет полосовой фильтр и проверяет RMS. Если канал слишком тихий, возвращает None."""
        # Полосовой фильтр
        channel_y = self.bandpass(channel_y, low=2500, high=5000)
        # RMS всего канала
        overall_rms = np.sqrt(np.mean(channel_y ** 2))
        if overall_rms < 0.005:
            return None
        return channel_y

    def analyze_channel_chunks(self, channel_y, sr):
        """ Анализ чанков внутри канала
        Разбивает канал на окна (chunk) и проверяет каждый chunk на наличие криков.
        Возвращает chunk_mask: список True/False по чанкам.
        Фон рассчитывается только по речевым фреймам (тишина исключается).
        """
        hop_samples = int(self.scream_hop_sec * sr)
        window_samples = int(self.scream_window_sec * sr)
        total_samples = len(channel_y)
        ignore_last_seconds = 1.5
        n_windows = max(1, (len(channel_y) - window_samples) // hop_samples + 1)
        chunk_mask = []
        for i in range(n_windows):
            start = i * hop_samples
            end = start + window_samples
            chunk = channel_y[start:end]

            # Время окна в секундах
            start_time = i * self.scream_hop_sec
            end_time = start_time + self.scream_window_sec
            # Пропуск последних окон
            if end_time > (total_samples / sr - ignore_last_seconds):
                chunk_mask.append(False)
                continue

            # 1. RMS и перевод в dB
            rms = self.compute_rms(chunk)
            rms_db = 20 * np.log10(rms + 1e-8)
            # 2. Фильтрация тишины
            self.silence_threshold_db = -45 # Порог тишины в дБ
            speech_frames = rms_db[rms_db > self.silence_threshold_db]
            # === 3. Бланк оценивается только по речи ===
            if len(speech_frames) > 0:
                # Берём 20% самых тихих среди РЕЧЕВЫХ фреймов
                threshold = np.percentile(speech_frames, 20)
                quiet_frames = speech_frames[speech_frames <= threshold]
            else:
                # fallback: если речи не обнаружено, берём старую схему
                threshold = np.percentile(rms_db, 20)
                quiet_frames = rms_db[rms_db <= threshold]
            # Средний "фон" (по речи)
            blank_mean = np.mean(quiet_frames) if len(quiet_frames) > 0 else np.percentile(rms_db, 10)
            blank_std = np.std(quiet_frames) if len(quiet_frames) > 1 else 1.0
            # Относительная громкость
            rms_rel = rms_db - blank_mean
            # === 4. Spectral flux (также на тех же чанках) ===
            flux = self.compute_flux(chunk)
            k = max(1, int(len(rms_db) * self.blank_pct))
            blank_flux = np.mean(np.sort(flux)[:k])
            flux_rel = flux / (blank_flux + 1e-8)
            # === 5. Флаги ===
            rms_flag = np.any(rms_rel > self.rms_multiplier * blank_std)
            flux_flag = np.any(flux_rel > self.flux_multiplier)
            high_flag = self.high_freq_energy(chunk) > self.high_freq_threshold
            stability_flag = np.std(rms_db) > 2.5  # нестабильность = характер крика
            # Итог: должны выполняться все 4 признака
            is_scream = rms_flag and flux_flag and high_flag and stability_flag
            chunk_mask.append(is_scream)
        return chunk_mask

    def merge_scream_intervals(self, chunk_mask, channel_y, sr):
        """ Слияние интервалов
        Преобразует маску True/False по чанкам в интервалы времени криков
        """
        merged_intervals = []
        current_start = None
        max_gap_chunks = int(self.max_silence_between_screams / self.scream_hop_sec)
        gap_count = 0
        for idx, val in enumerate(chunk_mask):
            if val:
                if current_start is None:
                    current_start = idx
                gap_count = 0
            else:
                if current_start is not None:
                    gap_count += 1
                    if gap_count > max_gap_chunks:
                        start_time = current_start * self.scream_hop_sec
                        end_time = (idx - gap_count) * self.scream_hop_sec
                        if end_time - start_time >= self.min_scream_sec:
                            merged_intervals.append((start_time, end_time))
                        current_start = None
                        gap_count = 0
        # Обработка последнего интервала
        if current_start is not None:
            start_time = current_start * self.scream_hop_sec
            end_time = min(
                (current_start + 1) * self.scream_hop_sec + self.scream_window_sec,
                len(channel_y) / sr
            )
            if end_time - start_time >= self.min_scream_sec:
                merged_intervals.append((start_time, end_time))
        return merged_intervals

    def detect_screams(self, audio_path):
        """
        Главная функция: детектирует крики во всех каналах аудио.
        Возвращает:
            all_channel_results: результаты по каждому каналу
            first_channel_intervals: интервалы криков первого канала
            first_channel_y: сигнал первого канала
            sr: частота дискретизации
            scream_ratio: доля времени с криком относительно общей длительности
        """
        # 1) Загрузка аудио
        y, sr = self.load_audio(audio_path)
        all_channel_results = []
        # Анализируем каждый канал
        for channel_idx in range(y.shape[0]):
            channel_y = y[channel_idx]
            # 2) Предобработка канала
            processed_channel = self.process_channel(channel_y)
            if processed_channel is None:
                # Канал слишком тихий → создаём пустой результат
                channel_result = {
                    'channel': channel_idx,
                    'chunk_mask': [],
                    'merged_intervals': [],
                    'scream_detected': False
                }
                all_channel_results.append(channel_result)
                continue
            # 3) Анализ чанков
            chunk_mask = self.analyze_channel_chunks(processed_channel, sr)
            # 4) Слияние интервалов криков
            merged_intervals = self.merge_scream_intervals(chunk_mask, processed_channel, sr)
            # Сохраняем результаты по каналу
            channel_result = {
                'channel': channel_idx,
                'chunk_mask': chunk_mask,
                'merged_intervals': merged_intervals,
                'scream_detected': len(merged_intervals) > 0
            }
            all_channel_results.append(channel_result)
        # Для обратной совместимости: первый канал
        first_channel_y = y[0] if y.shape[0] > 0 else np.array([])
        first_channel_mask = all_channel_results[0]['chunk_mask'] if all_channel_results else []
        first_channel_intervals = all_channel_results[0]['merged_intervals'] if all_channel_results else []
        # Вычисляем коэффициент крика
        total_duration = len(y[0]) / sr if y.shape[0] > 0 else 0
        total_scream_time = sum(
            (end - start)
            for channel in all_channel_results
            for start, end in channel['merged_intervals']
        )
        scream_ratio = total_scream_time / total_duration if total_duration > 0 else 0
        return all_channel_results, first_channel_intervals, first_channel_y, sr, scream_ratio

    def detect_overlap(self, audio_path):
        '''функия определения перебиваний'''
        y, sr = self.load_audio(audio_path)
        # делаем в стерео из моно (в проде не надо!)
        if y.ndim == 1:
            y = np.vstack([y, y])

        #  ФИЛЬТРАЦИЯ - оставляем только речевой диапазон
        y_filtered = np.zeros_like(y)
        for ch in range(y.shape[0]):
            y_filtered[ch] = self.bandpass(y[ch], low=300, high=4500)

        chunk_samples = int(self.overlap_chunk_sec * sr) # размер чанка
        n_chunks = y.shape[1] // chunk_samples + 1 # количество чанков в аудио
        # анализ активности голоса в каждом чанке
        overlap_mask = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            voices_active = 0
            for ch in range(y.shape[0]):
                chunk = y_filtered[ch, start:end] # заменяем здесь y на фильтрованный y_filtered
                rms = np.sqrt(np.mean(chunk ** 2))  # вычисляет RMS (среднеквадратичное значение) - меру громкости
                rms_db = 20 * np.log10(rms + 1e-10)
                if rms_db > self.rms_threshold_overlap: # сравниваем с порогом
                    voices_active += 1
            '''массив из TRUE/False равный количеству чанков'''
            overlap_mask.append(voices_active >= 2) # определяем перекрытие если активности на двух каналах

        # Объединение интервалов
        merged_intervals = []
        current_start = None
        max_gap_chunks = 1 # максимально допустимый разрыв между чанками (не больше одного)
        gap = 0 # счетчик последовательных чанков
        for idx, val in enumerate(overlap_mask):
            if val:  # Текущий чанк - перекрытие
                if current_start is None:
                    current_start = idx
                gap = 0
            else:   # Текущий чанк - нет перекрытия
                if current_start is not None:
                    gap += 1
                    if gap > max_gap_chunks:    # Промежуток слишком велик
                        start_time = current_start * self.overlap_chunk_sec
                        end_time = (idx - gap + 1) * self.overlap_chunk_sec
                        if end_time - start_time >= self.min_overlap_sec:   # отсекаем участи где минимальный интервал перебиваний меньше
                            merged_intervals.append((start_time, end_time))
                        current_start = None
                        gap = 0
        # Обработка последнего интервала
        if current_start is not None:
            start_time = current_start * self.overlap_chunk_sec
            end_time = n_chunks * self.overlap_chunk_sec
            if end_time - start_time >= self.min_overlap_sec:
                merged_intervals.append((start_time, end_time))
        return merged_intervals

    def analyze_audio(self, audio_path):
        '''анализ аудио'''
        all_channel_results, scream_intervals, y, sr, scream_ratio  = self.detect_screams(audio_path)
        overlap_intervals = self.detect_overlap(audio_path)

        # Определяем общее наличие криков (есть ли крики хотя бы в одном канале)
        overall_scream_detected = any(channel['scream_detected'] for channel in all_channel_results)

        # Собираем все интервалы криков со всех каналов
        all_scream_intervals = []
        for channel_result in all_channel_results:
            all_scream_intervals.extend([(channel_result['channel'], round(start,1), round(end,1))
                                         for start, end in channel_result['merged_intervals']])

        total_time = len(y) / sr
        overlap_duration = sum(end - start for start, end in overlap_intervals)
        overlap_ratio = overlap_duration / total_time if total_time > 0 else 0

        # Округляем интервалы перебиваний
        rounded_overlap_intervals = [(round(start, 1), round(end, 1)) for start, end in overlap_intervals]

        # 1. Исправляем нумерацию каналов (делаем с 1)
        scream_intervals_fixed = [
            [channel + 1, round(start, 1), round(end, 1)]
            for channel, start, end in all_scream_intervals
        ]

        # 2. Округляем интервалы в channels_analysis
        rounded_channels_analysis = []
        for channel in all_channel_results:
            rounded_intervals = [(round(start, 1), round(end, 1)) for start, end in channel['merged_intervals']]
            rounded_channels_analysis.append({
                "channel": channel["channel"] + 1,  # Нумерация с 1
                "scream_detected": channel["scream_detected"],
                "intervals_count": len(rounded_intervals),
                "intervals": rounded_intervals
            })

        results = {
            "total_duration": round(total_time, 2),
            "scream_detected": overall_scream_detected,       # наличие крика
            "scream_ratio": round(scream_ratio, 3),           # коэф крика
            "scream_intervals": scream_intervals_fixed,  # список кортежей (канал, начало, конец)
            "overlap_detected": len(overlap_intervals) > 0,   # наличие перебиваний
            "overlap_ratio": round(overlap_ratio, 3),         # коэф перебиваний
            "overlap_intervals": rounded_overlap_intervals,
            "channels_analysis": rounded_channels_analysis     # детальная информация по каждому каналу(крики)
        }
        return results

    def analyze_audio_complete(self, filename):
        """Полный анализ аудио с постобработкой"""
        start_time = time.perf_counter()
        try:

            #print(f"detector_params: {self._get_current_params()}")
            # Базовый анализ
            basic_results = self.analyze_audio(audio_path=filename)
            filename = filename.split('.')[-2].split('/')[-1]
            # Добавляем метаинформацию
            time_func = round(time.perf_counter() - start_time, 1)
            complete_results = {
                "filename": filename,
                **basic_results,
                "time_func_analyze": time_func,
                "config_mode": self.config_mode,
                "detector_params": self._get_current_params(),
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            return complete_results

        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error_message": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }

    def _get_current_params(self):
        """Возвращает текущие параметры детектора"""
        all_param_keys = [
            # Системные
            "config_mode"
            # Крики (настраиваемые)
            "scream_window_sec", "scream_hop_sec", "blank_pct",
            "rms_multiplier", "flux_multiplier", "min_scream_sec",
            "max_silence_between_screams", "high_freq_threshold",

            # Перебивания (настраиваемые)
            "overlap_chunk_sec", "min_overlap_sec", "rms_threshold_overlap",
        ]
        return {key: getattr(self, key) for key in all_param_keys if hasattr(self, key)}


    def plot_screams_and_overlap(self, y, scream_intervals, overlap_intervals, plot_path="audio_events.png"):
        '''визуализация'''
        if not scream_intervals and not overlap_intervals:
            print("Событий не обнаружено, график не строится")
            return
        times = np.arange(len(y)) / self.sr
        plt.figure(figsize=(15, 4))
        plt.plot(times, y, color='gray', alpha=0.6)
        for start, end in scream_intervals:
            plt.axvspan(start, end, color='red', alpha=0.4)
        for start, end in overlap_intervals:
            plt.axvspan(start, end, color='blue', alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Audio waveform with screams (red) and overlaps (blue)")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"График событий сохранен: {plot_path}")


def writer_log(data):
    try:
        with open('json_file.json', 'a', encoding='utf-8') as f:
            log_line = json.dumps(data, ensure_ascii=False)
            f.write(f"{log_line}\n")
    except Exception as e:
        print(f"Ошибка записи лога {e}")


def main():
    detector = AudioEventDetector(config_mode="lite")
    # запуск вручную
    folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\temp_audio\new_interruptions_2_lite"
    folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\tests"
    folder_path = r'C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\крики'
    folder_path = r'C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\гудки'
    folder_path = r'C:\Users\beginin-ov\Projects\Local\Аудио распознание\temp_audio\new_scream_2'
    # Все файлы
    all_files = glob.glob(os.path.join(folder_path, "*"))
    '''all_files = []'''
    #print(all_files)
    # for filename in all_files[:]:
    #     try:
    #         #print(all_files)
    #         #print(filename.split('.')[-2].split('\\')[-1])
    #         #print(filename, filename.split('.')[-2].split("\\")[-1])
    #         print(filename)
    #         result = detector.analyze_audio_complete(filename=filename)
    #         filename_short = filename.split('.')[-2].split('\\')[-1]
    #         #print(f"{filename.split('.')[-2].split('\\')[-1]}: {result}")
    #         #print( f"{filename.split('.')[-2].split('\\')[-1]}: {result["overlap_detected"]}  {result["overlap_intervals"]}")
    #         print(f"{filename_short}: {result["scream_detected"]}  {result["scream_intervals"]}")
    #         result = {**result}
    #         result.update({"filename": filename_short})
    #         writer_log(data=result)
    #     except Exception as e:
    #         print(e)
    #         continue






def test_all():
    '''функция для тестирования различных параметров для определения крика(передать как список из словарей)'''
    # запуск вручную
    folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\temp_audio\new_scream"
    #folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\scream_audio"
    # Все файлы
    all_files = glob.glob(os.path.join(folder_path, "*"))

    default_params_norm = {
        "rms_multiplier": 5.0,  # -0.4% от предыдущего
        "flux_multiplier": 5.0,  # -0.4% от предыдущего
        "high_freq_threshold": 0.06,  # -0.3% от предыдущего
        "min_scream_sec": 1.1,  # -0.9% от предыдущего
        "scream_window_sec": 2.0,
        "scream_hop_sec": 0.4,
        "blank_pct": 0.45,  # -2.2% от предыдущего
    }
    for filename in all_files[13:]:
        try:
            #print(all_files)
            #print(filename.split('.')[-2].split('\\')[-1])
            #print(filename, filename.split('.')[-2].split("\\")[-1])

            result = analyze_all(filename=filename, default_params=default_params_norm)
            print(f"{filename.split('.')[-2].split('\\')[-1]}: {result["scream_detected"]}  {result["scream_intervals"]}")
            #result["filename"]=filename.split('.')[-2].split('\\')[-1]

        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    main()
def test_2():
    # запуск вручную
    folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\tests"
    #folder_path = r"C:\Users\beginin-ov\Projects\Local\Аудио распознание\отчёт\scream_audio"

    # Все файлы
    #all_files = glob.glob(os.path.join(folder_path, "*"))
    all_files =[
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\01.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\avtootvet1.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\avtootvet2.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\avtootvet3.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\avtootvet4.mp3',
     r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\high.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\high_client.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\hith_operator.mp3',
     r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\shum.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\simple.mp3',
     # r'C:\\Users\\beginin-ov\\Projects\\Local\\Аудио распознание\\отчёт\\tests\\simple2.mp3'
        ]

    default_params = {
        "rms_multiplier": 5.0,  # УВЕЛИЧЕНО: требует большей громкости
        "flux_multiplier": 5.0,  # УВЕЛИЧЕНО: требует более резких изменений
        "high_freq_threshold": 0.055,  # УВЕЛИЧЕНО: требует больше высокой энергии
        "min_scream_sec": 0.8,  # УВЕЛИЧЕНО: игнорирует короткие звуки
        "scream_window_sec": 2.0,
        "scream_hop_sec": 0.5,
        "blank_pct": 0.5,  # УВЕЛИЧЕНО: более точный расчет шума
    }
    # print(all_files)
    param_configs = [
        # 1. ГРАНИЦА ПО RMS (5.1 - между 5.0 и 5.2)
        {
            "rms_multiplier": 5.1,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.06,
            "min_scream_sec": 1.1,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 2. ГРАНИЦА ПО HIGH_FREQ (0.061 - между 0.06 и 0.062)
        {
            "rms_multiplier": 5.0,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.061,
            "min_scream_sec": 1.1,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 3. ГРАНИЦА ПО MIN_SCREAM_SEC (1.15 - между 1.1 и 1.2)
        {
            "rms_multiplier": 5.0,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.06,
            "min_scream_sec": 1.15,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 4. КОМБИНАЦИЯ RMS + HIGH_FREQ (оба на границе)
        {
            "rms_multiplier": 5.1,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.061,
            "min_scream_sec": 1.1,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 5. КОМБИНАЦИЯ ВСЕХ ГРАНИЦ
        {
            "rms_multiplier": 5.05,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.0605,
            "min_scream_sec": 1.12,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 6. ПРОВЕРКА FLUX НА ГРАНИЦЕ (5.25)
        {
            "rms_multiplier": 5.0,
            "flux_multiplier": 5.25,
            "high_freq_threshold": 0.06,
            "min_scream_sec": 1.1,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 7. СЛЕГКА СТРОЖЕ ПО ВСЕМ ПАРАМЕТРАМ
        {
            "rms_multiplier": 5.08,
            "flux_multiplier": 5.1,
            "high_freq_threshold": 0.0608,
            "min_scream_sec": 1.13,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.46,
        },
        # 8. МАКСИМАЛЬНО БЛИЗКО К ГРАНИЦЕ (почти срабатывает)
        {
            "rms_multiplier": 5.02,
            "flux_multiplier": 5.02,
            "high_freq_threshold": 0.0602,
            "min_scream_sec": 1.11,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.46,
        },
        # 9. АКЦЕНТ НА RMS И ДЛИТЕЛЬНОСТЬ
        {
            "rms_multiplier": 5.09,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.06,
            "min_scream_sec": 1.14,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        },
        # 10. АКЦЕНТ НА HIGH_FREQ И ДЛИТЕЛЬНОСТЬ
        {
            "rms_multiplier": 5.0,
            "flux_multiplier": 5.0,
            "high_freq_threshold": 0.0609,
            "min_scream_sec": 1.14,
            "scream_window_sec": 2.0,
            "scream_hop_sec": 0.4,
            "blank_pct": 0.45,
        }
    ]
    for filename in all_files[:]:
        try:
            #print(all_files)
            #print(filename.split('.')[-2].split('\\')[-1])
            #print(filename, filename.split('.')[-2].split("\\")[-1])
            for i, params in enumerate(param_configs, 1):
                try:
                    print(f"Конфигурация {i}: {params}")
                    result = analyze_all(filename=filename, default_params=params)
                    print(f"{filename.split('.')[-2].split('\\')[-1]}: {result["scream_detected"]}  {result["scream_intervals"]}")
                    #result["filename"]=filename.split('.')[-2].split('\\')[-1]
                except Exception as e:
                    continue
        except Exception as e:
            print(e)
            continue
    #print(all_files)