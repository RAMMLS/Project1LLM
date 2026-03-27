import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # Базовые настройки
        self.DATASET_ID = os.getenv("DATASET_ID", "iqmansingh/guns-knives-object-detection")
        self.EXTRACT_DIR = os.getenv("EXTRACT_DIR", "./data/yolo_dataset")
        
        # Настройки YOLO
        self.MODEL_TYPE = os.getenv("MODEL_TYPE", "yolo11n.pt")
        self.EPOCHS = int(os.getenv("EPOCHS", "1"))
        self.FRACTION = float(os.getenv("FRACTION", "0.2"))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
        self.DEVICE = os.getenv("DEVICE", "cpu")  # '0' for GPU или 'cpu'
        
        # Пути
        self.RUNS_DIR = os.getenv("RUNS_DIR", "./runs")

    def update(self, **kwargs):
        """Обновление настроек 'на лету'."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

config = Config()
