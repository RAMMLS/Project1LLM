import os
import shutil
import stat
import threading
from contextlib import redirect_stdout, redirect_stderr
import kagglehub
import yaml
import logging
from typing import Dict, Any, List, Optional
from models.vision_yolo import YOLOVisionModel
from config import config

logger = logging.getLogger(__name__)

class _TrainingLogWriter:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""

    def write(self, value: str) -> int:
        if not value:
            return 0
        # Заменяем \r на \n, чтобы ловить прогресс-бары (tqdm и т.д.)
        value = value.replace("\r", "\n")
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.callback(line)
        return len(value)

    def flush(self) -> None:
        line = self.buffer.strip()
        if line:
            self.callback(line)
        self.buffer = ""

class CVService:
    """
    Сервис для управления данными машинного зрения (CV) и процессом обучения.
    Портировано и адаптировано из Jupyter-блокнота.
    """
    def __init__(self):
        self.model: Optional[YOLOVisionModel] = None
        self.dataset_path: Optional[str] = None
        self.prepared_dataset_id: Optional[str] = None
        self.training_logs: List[str] = []
        self.training_active = False
        self.training_phase = "idle"
        self.training_lock = threading.Lock()
    
    def _remove_readonly(self, func, path, exc_info):
        error = exc_info[1]
        if isinstance(error, FileNotFoundError) or not os.path.exists(path):
            return
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except FileNotFoundError:
            return
        except Exception:
            raise error

    def _clear_target_dir(self, target_dir: str) -> None:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, onerror=self._remove_readonly)

    def _has_dataset_structure(self, dataset_path: str) -> bool:
        if not os.path.isdir(dataset_path):
            return False

        for root, dirs, _ in os.walk(dataset_path):
            if "train" not in dirs:
                continue
            train_images = os.path.join(root, "train", "images")
            valid_images = os.path.join(root, "valid", "images")
            test_images = os.path.join(root, "test", "images")
            if os.path.isdir(train_images) and (os.path.isdir(valid_images) or os.path.isdir(test_images)):
                return True
        return False

    def count_images(self, directory: str) -> int:
        """Подсчет количества изображений в директории (из блокнота)."""
        count = 0
        if not os.path.exists(directory):
            return 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpeg", ".jpg")):
                    count += 1
        return count

    def get_dataset_stats(self) -> Dict[str, int]:
        """Получение статистики текущего датасета."""
        if not self.dataset_path:
            return {"error": "Датасет не загружен"}
        
        # Пытаемся найти папки train и valid внутри
        # В блокноте они могли быть вложены: guns-knives-yolo/guns-knives-yolo/train
        potential_train = os.path.join(self.dataset_path, "train")
        potential_val = os.path.join(self.dataset_path, "valid")
        
        # Если не нашли в корне, ищем глубже (как в блокноте)
        if not os.path.exists(potential_train):
            for root, dirs, _ in os.walk(self.dataset_path):
                if "train" in dirs:
                    potential_train = os.path.join(root, "train")
                if "valid" in dirs:
                    potential_val = os.path.join(root, "valid")

        return {
            "train_count": self.count_images(potential_train),
            "val_count": self.count_images(potential_val),
            "path": self.dataset_path
        }

    def reset_training_logs(self) -> None:
        with self.training_lock:
            self.training_logs = []
            self.training_active = False
            self.training_phase = "idle"

    def start_training_session(self, dataset_id: str, epochs: Optional[int], device: Optional[str], fraction: Optional[float]) -> None:
        with self.training_lock:
            if self.training_active:
                raise RuntimeError("Обучение YOLO уже запущено. Дождитесь завершения текущего процесса.")
            self.training_logs = []
            self.training_active = True
            self.training_phase = "queued"
        self.append_training_log("=== YOLO JOB ACCEPTED ===")
        self.append_training_log(f"dataset_id={dataset_id}")
        self.append_training_log(f"epochs={epochs or config.EPOCHS}, device={device or config.DEVICE}, fraction={fraction if fraction is not None else config.FRACTION}")
        self.append_training_log("Ожидание старта фоновой задачи...")

    def append_training_log(self, message: str) -> None:
        clean_message = message.strip()
        if not clean_message:
            return
        with self.training_lock:
            self.training_logs.append(clean_message)
            if len(self.training_logs) > 1000:
                self.training_logs = self.training_logs[-1000:]

    def set_training_active(self, active: bool) -> None:
        with self.training_lock:
            self.training_active = active

    def set_training_phase(self, phase: str) -> None:
        with self.training_lock:
            self.training_phase = phase

    def get_training_state(self, from_index: int = 0) -> Dict[str, Any]:
        with self.training_lock:
            logs = self.training_logs[from_index:]
            next_index = len(self.training_logs)
            is_active = self.training_active
            phase = self.training_phase
        return {
            "logs": logs,
            "next_index": next_index,
            "is_active": is_active,
            "phase": phase
        }

    def prepare_dataset(self, dataset_id: str = config.DATASET_ID) -> str:
        """
        Скачивает датасет с Kaggle и подготавливает структуру.
        :param dataset_id: ID датасета на Kagglehub.
        :return: Путь к подготовленному датасету.
        """
        self.append_training_log(f"Запуск подготовки датасета: {dataset_id}...")
        try:
            self.set_training_phase("preparing_dataset")
            self.append_training_log("Шаг 1: Скачивание датасета с Kagglehub (может занять время)...")
            download_path = kagglehub.dataset_download(dataset_id)
            self.append_training_log(f"Датасет скачан во временную директорию: {download_path}")

            target_dir = os.path.abspath(config.EXTRACT_DIR)
            if self.prepared_dataset_id == dataset_id and self._has_dataset_structure(target_dir):
                self.dataset_path = target_dir
                self.append_training_log(f"Локальная копия датасета уже готова: {target_dir}")
                return target_dir

            self.append_training_log(f"Шаг 2: Копирование файлов в {target_dir}...")
            self._clear_target_dir(target_dir)
            
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            shutil.copytree(download_path, target_dir, copy_function=shutil.copyfile)
            
            self.dataset_path = target_dir
            self.prepared_dataset_id = dataset_id
            self.append_training_log(f"Датасет успешно подготовлен в: {target_dir}")
            return target_dir
        except Exception as e:
            self.append_training_log(f"Критическая ошибка при подготовке датасета: {e}")
            raise

    def create_data_yaml(self, dataset_path: str, classes: Dict[int, str]) -> str:
        """Создает файл data.yaml для YOLO, находя правильные пути."""
        # Пытаемся найти, где реально лежат train и val/test
        train_path = "train/images"
        val_path = "valid/images"
        
        # Поиск реальных путей (учет вложенности Kaggle)
        for root, dirs, _ in os.walk(dataset_path):
            if "train" in dirs and "images" in os.listdir(os.path.join(root, "train")):
                train_path = os.path.relpath(os.path.join(root, "train/images"), dataset_path)
            if "valid" in dirs and "images" in os.listdir(os.path.join(root, "valid")):
                val_path = os.path.relpath(os.path.join(root, "valid/images"), dataset_path)
            elif "test" in dirs and "images" in os.listdir(os.path.join(root, "test")):
                val_path = os.path.relpath(os.path.join(root, "test/images"), dataset_path)
        
        yaml_data = {
            'path': os.path.abspath(dataset_path),
            'train': train_path.replace("\\", "/"),
            'val': val_path.replace("\\", "/"),
            'names': classes
        }
        
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            return yaml_path
        except Exception as e:
            logger.error(f"Ошибка при создании data.yaml: {e}")
            raise

    def get_random_samples(self, subset: str = "train", num_samples: int = 5) -> List[str]:
        """Возвращает пути к случайным изображениям из датасета (аналог визуализации из блокнота)."""
        if not self.dataset_path:
            return []
        
        subset_path = os.path.join(self.dataset_path, subset, "images")
        if not os.path.exists(subset_path):
            # Пробуем найти вложенный путь
            for root, dirs, _ in os.walk(self.dataset_path):
                if subset in dirs:
                    subset_path = os.path.join(root, subset, "images")
                    break
        
        if not os.path.exists(subset_path):
            return []

        import random
        all_images = [os.path.join(subset_path, f) for f in os.listdir(subset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        return random.sample(all_images, min(len(all_images), num_samples))

    def start_training(self, dataset_id: Optional[str] = None, epochs: Optional[int] = None, device: Optional[str] = None, fraction: Optional[float] = None) -> Dict[str, Any]:
        """Запуск обучения с использованием текущих настроек."""
        self.set_training_active(True)
        dataset_id = dataset_id or config.DATASET_ID
        epochs = epochs or config.EPOCHS
        device = device or config.DEVICE
        fraction = fraction if fraction is not None else config.FRACTION

        log_writer = _TrainingLogWriter(self.append_training_log)
        try:
            with redirect_stdout(log_writer), redirect_stderr(log_writer):
                self.append_training_log("=== BACKGROUND TASK STARTED ===")
                self.append_training_log(f"Подготовка датасета: {dataset_id}")
                self.prepare_dataset(dataset_id=dataset_id)

                yaml_path = os.path.join(self.dataset_path, 'data.yaml')
                if not os.path.exists(yaml_path):
                    self.create_data_yaml(self.dataset_path, {0: 'knife', 1: 'pistol'})
                self.append_training_log(f"data.yaml: {yaml_path}")
                self.append_training_log("=== DATASET READY ===")

                if not self.model:
                    self.model = YOLOVisionModel(config.MODEL_TYPE)
                self.append_training_log(f"Модель: {config.MODEL_TYPE}")
                self.append_training_log("=== ULTRALYTICS TRAIN START ===")
                self.set_training_phase("training")

                results = self.model.train(
                    data_yaml_path=yaml_path,
                    epochs=epochs,
                    device=device,
                    fraction=fraction,
                    project=config.RUNS_DIR,
                    name="yolo_train_run"
                )
            log_writer.flush()
            self.set_training_phase("done")
            self.append_training_log("Обучение YOLO успешно завершено.")
            return results
        except Exception as e:
            log_writer.flush()
            self.set_training_phase("error")
            self.append_training_log(f"Ошибка обучения YOLO: {e}")
            raise
        finally:
            self.set_training_active(False)
