import os
import logging
from typing import Optional, Dict, Any

# Перенаправляем конфиг-директории YOLO в папку проекта, чтобы избежать PermissionError
# Это нужно сделать ДО импорта ultralytics
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["ULTRALYTICS_CONFIG_DIR"] = os.path.join(project_root, ".ultralytics")
os.makedirs(os.environ["ULTRALYTICS_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO
import torch

# Настройка логирования
logger = logging.getLogger(__name__)

class YOLOVisionModel:
    """
    Класс-обертка для модели YOLO (Ultralytics).
    Позволяет инициализировать модель, загружать веса, обучать и делать предсказания.
    """
    def __init__(self, model_name: str = 'yolo11n.pt'):
        """
        Инициализация YOLO.
        :param model_name: Название модели или путь к весам (.pt).
        """
        try:
            self.model = YOLO(model_name)
            logger.info(f"YOLO модель {model_name} успешно инициализирована.")
        except Exception as e:
            logger.error(f"Ошибка при инициализации YOLO: {e}")
            raise

    def train(self, data_yaml_path: str, epochs: int = 1, device: str = 'cpu', fraction: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Запуск процесса обучения.
        :param data_yaml_path: Путь к data.yaml с конфигурацией датасета.
        :param epochs: Количество эпох.
        :param device: Устройство ('cpu' или '0', '1' для GPU).
        :param fraction: Доля датасета для обучения (от 0.0 до 1.0).
        :param kwargs: Дополнительные параметры для YOLO.train().
        :return: Результаты обучения.
        """
        logger.info(f"Начинаем обучение на {epochs} эпох на устройстве {device} с fraction={fraction}...")
        try:
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                device=device,
                fraction=fraction,
                **kwargs
            )
            logger.info("Обучение завершено успешно.")
            return results
        except Exception as e:
            logger.error(f"Ошибка в процессе обучения: {e}")
            raise

    def predict(self, source: str, conf: float = 0.25, **kwargs) -> Any:
        """
        Выполнение детекции на изображении или видео.
        :param source: Путь к файлу или URL.
        :param conf: Порог уверенности.
        :return: Список результатов детекции.
        """
        logger.info(f"Выполнение детекции для: {source}")
        try:
            results = self.model.predict(source=source, conf=conf, **kwargs)
            return results
        except Exception as e:
            logger.error(f"Ошибка при детекции: {e}")
            raise

    def export(self, format: str = 'onnx'):
        """Экспорт модели в другой формат."""
        logger.info(f"Экспорт модели в формат {format}...")
        return self.model.export(format=format)
