# Torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Визуализация
import seaborn as sns

# Метрики
from sklearn.metrics import accuracy_score
import pandas as pd
# Остальное
from tqdm.notebook import tqdm
from IPython.display import clear_output

# Configuration
from image_classification.data import *

class ImageClassification(nn.Module):
    def __init__(self, model, name='Model', optimizer=None, loss_fn=None, metric=None, scheduler=None, save_dir="./models"):
        super().__init__()

        # Название модели
        self.name = name

        # Путь для сохранения модели
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.path = f"{save_dir}/{self.name}/Best_{self.name}.pth"

        # Оптимизатор
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.__optimizer = optimizer

        # Функция потерь
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.__loss_fn = loss_fn

        # Шедулер
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.scheduler = scheduler

        # Метрика
        if metric is None:
            metric = accuracy_score
        self.__metric = metric

        # Переносим модель на устройство
        self.__model = model.to(device)

        # История качества
        self.__train_loss_history, self.__valid_loss_history = [], []
        self.__train_score_history, self.__valid_score_history = [], []

        # Лучшие значения
        self.best_epoch, self.best_score, self.best_loss = 0, 0, float('inf')

        # Флаг для остановки обучения
        self.stop_fiting = False

    def forward(self, x):
        return self.__model(x)

    def run_epoch(self, data_loader, mode='train'):
        if mode == 'train':
            self.__model.train()
        elif mode == 'eval':
            self.__model.eval()
        else:
            raise ValueError("Mode должен быть 'train' или 'eval'.")

        count = 0
        total_loss = 0
        labels_true, labels_pred = [], []

        torch.set_grad_enabled(mode == 'train')

        progress_desc = 'Training' if mode == 'train' else 'Evaluating'
        progress_bar = tqdm(data_loader, desc=progress_desc)

        try:
            for images, labels in progress_bar:
                if mode == 'train':
                    self.__optimizer.zero_grad()

                images, labels = images.to(device), labels.to(device)
                output = self.__model(images)
                if hasattr(output, 'logits'):
                    output = output.logits
                loss = self.__loss_fn(output, labels)

                if mode == 'train':
                    loss.backward()
                    self.__optimizer.step()

                total_loss += loss.item()
                labels_true.extend(labels.cpu().numpy().flatten())
                labels_pred.extend(output.argmax(dim=1).cpu().numpy().flatten())

                count += 1

                current_loss = total_loss / count
                current_score = self.__metric(labels_true, labels_pred)
                progress_bar.set_postfix(
                    **{
                        self.__loss_fn.__class__.__name__: f"{current_loss:.4f}",
                        self.__metric.__name__: f"{current_score:.4f}",
                    }
                )

        except KeyboardInterrupt:
            self.stop_fiting = True
            print(f"\n{progress_desc} прервано пользователем. Завершаем текущую эпоху...")

        return total_loss / count, self.__metric(labels_true, labels_pred)

    def plot_stats(self):
        plt.figure(figsize=(16, 8))
        epochs = range(1, len(self.__train_loss_history) + 1)

        plt.subplot(1, 2, 1)
        sns.lineplot(x=epochs, y=self.__train_loss_history, label='Train Loss', linestyle='--', marker='o', color='#1f77b4', linewidth=3)
        sns.lineplot(x=epochs, y=self.__valid_loss_history, label='Valid Loss', linestyle='-', marker='o', color='#bc4b51', linewidth=3)
        plt.title(f'{self.name} - {self.__loss_fn.__class__.__name__}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.lineplot(x=epochs, y=self.__train_score_history, label=f'Train {self.__metric.__name__}', linestyle='--', marker='o', linewidth=3)
        sns.lineplot(x=epochs, y=self.__valid_score_history, label=f'Valid {self.__metric.__name__}', linestyle='-', marker='o', linewidth=3)
        plt.title(f'{self.name} - {self.__metric.__name__}')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def fit(self, train_loader, valid_loader, num_epochs, visualize=True, use_best_model=True, min_loss=False, eps=0.001):
        sns.set_style('whitegrid')
        sns.set_palette('Set2')
        

        # Создаем подпапку для сохранения моделей, если она не существует
        model_dir = os.path.join("./models", self.name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Создаем папку для эпох, если она не существует
        epochs_dir = os.path.join(model_dir, "epochs")
        if not os.path.exists(epochs_dir):
            os.makedirs(epochs_dir)
        

        history = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "train_score": [],
            "valid_score": []
        }

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history) + 1})\n")
            train_loss, train_score = self.run_epoch(train_loader, mode='train')
            valid_loss, valid_score = self.run_epoch(valid_loader, mode='eval')

            self.__train_loss_history.append(train_loss)
            self.__valid_loss_history.append(valid_loss)
            self.__train_score_history.append(train_score)
            self.__valid_score_history.append(valid_score)

            clear_output()

            print(f"Epoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history)})\n")
            print(f'Loss: {self.__loss_fn.__class__.__name__}')
            print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")

            print(f"Score: {self.__metric.__name__}")
            print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")

            if (self.best_score is None or (valid_score - self.best_score > eps) or min_loss) and (
                    not min_loss or valid_loss < self.best_loss) and not self.stop_fiting:
                print("(Model saved)")
                self.best_epoch, self.best_score, self.best_loss = len(self.__train_loss_history), valid_score, valid_loss
                self.save_model()

            # Сохраняем модель для текущей эпохи
            epoch_model_path = os.path.join(epochs_dir, f"epoch_{epoch}.pth")
            torch.save(self.__model.state_dict(), epoch_model_path)
            print(f"Model for epoch {epoch} saved at {epoch_model_path}")

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["train_score"].append(train_score)
            history["valid_score"].append(valid_score)

            # Визуализация результатов после второй эпохи
            if len(self.__train_loss_history) > 1:
                if visualize:
                    self.plot_stats()

                print(f"Best valid score: {self.best_score:.4f} ({self.best_epoch} epoch)\n")

            # Проверяем флаг остановки обучения
            if self.stop_fiting:
                print("Обучение остановлено пользователем после текущей эпохи.")

                self.stop_fiting = False
                break

        history_df = pd.DataFrame(history)
        history_csv_path = os.path.join("./models", self.name, "training_history.csv")
        history_df.to_csv(history_csv_path, index=False)


        # Загружаем лучшие веса модели
        if use_best_model and os.path.exists(self.path):
            # Losses
            self.__train_loss_history = self.__train_loss_history[:self.best_epoch]
            self.__valid_loss_history = self.__valid_loss_history[:self.best_epoch]
            # Scores
            self.__train_score_history = self.__train_score_history[:self.best_epoch]
            self.__valid_score_history = self.__valid_score_history[:self.best_epoch]

            # Load best model
            self.load()

    @torch.inference_mode()
    def predict_proba(self, inputs, batch_size=50, progress_bar=True):
        # Обработка одного изображения
        if isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
            return self.__model(inputs.unsqueeze(0).to(device))[0].tolist()

        # Определяем, является ли входной списоком или датасетом
        if isinstance(inputs, (list, ImageDataset)):
            predictions = []
            data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)

            if progress_bar:
                data_loader = tqdm(data_loader, desc="Predicting")

            # Итерация по батчам
            for batch in data_loader:
                batch_predictions = self.__model(batch.to(device))
                predictions.append(batch_predictions)

            return torch.cat(predictions, dim=0).tolist()

        # Если формат данных неизвестен
        raise ValueError("Unsupported input type. Expected single tensor, list of tensors, or Dataset.")

    def predict(self, inputs, *args, **kwargs):
        return np.argmax(self.predict_proba(inputs, *args, **kwargs), axis=1)

    def save_model(self):
        torch.save(self.__model.state_dict(), self.path)

    def load(self):
        self.__model.eval()
        self.__model.load_state_dict(torch.load(self.path, map_location=device))
