import torch
import torch.nn as nn

class MathModel(nn.Module):
    """
    Модуль В: Математика и Последовательности.
    Это наша песочница для многослойных персептронов (MLP) и рекуррентных сеток (RNN).
    Тут можно налету менять функции активации (ReLU, Tanh, Sigmoid) и собирать слои как конструктор Lego.
    """
    def __init__(self, input_dim=10, hidden_layers=[64, 32], output_dim=1, activation='relu', model_type='mlp'):
        super(MathModel, self).__init__()
        self.model_type = model_type.lower()
        self.activation_str = activation.lower()
        self.activation_fn = self._get_activation()
        
        if self.model_type == 'mlp':
            # Собираем классический персептрон в цикле
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(self.activation_fn)
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
        elif self.model_type == 'rnn':
            # Простенькая RNN-ка для работы с временными рядами (типа котировок)
            hidden_size = hidden_layers[0] if hidden_layers else 32
            num_layers = len(hidden_layers) if hidden_layers else 1
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_dim)
        else:
            raise ValueError(f"Бро, такой модели нет: {model_type}")

    def _get_activation(self):
        # Выбираем функцию активации по строке из API
        if self.activation_str == 'relu':
            return nn.ReLU()
        elif self.activation_str == 'tanh':
            return nn.Tanh()
        elif self.activation_str == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Неизвестная активация: {self.activation_str}")

    def forward(self, x):
        if self.model_type == 'mlp':
            return self.network(x)
        elif self.model_type == 'rnn':
            # Если это RNN, то на входе ждем тензор: (размер_батча, длина_последовательности, размер_фичи)
            out, _ = self.rnn(x)
            # Берем только последний выход (типа предсказание на следующий день)
            out = out[:, -1, :]
            return self.fc(out)
