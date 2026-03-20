import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumModel(nn.Module):
    """
    Модуль Б: 1D Спектроскопия.
    Тут у нас одномерная сверточная сетка (1D-CNN) для данных типа ИК-Фурье спектроскопии.
    Через API можно крутить размер окна (kernel_size) и дропаут (dropout), 
    чтобы тестить насколько локальные признаки мы ловим и бороться с переобучением.
    """
    def __init__(self, input_channels=1, num_classes=5, kernel_size=3, dropout_rate=0.5):
        super(SpectrumModel, self).__init__()
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Одномерные сверточные слои (скользим по спектру)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        # Полносвязные слои для классификации
        # Адаптивный пулинг нужен, чтобы не париться с точной длиной входного спектра
        # Он всегда сожмет фичи до размера 16, а потом мы это вытянем в вектор
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.fc1 = nn.Linear(32 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x на входе должен быть размера: (размер_батча, каналы, длина_спектра)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Вытягиваем тензор в колбасу (сплющиваем)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def update_dropout(self, new_dropout_rate):
        """Метод чтобы на лету менять вероятность выбивания нейронов (дропаут)"""
        self.dropout_rate = new_dropout_rate
        self.dropout.p = new_dropout_rate
