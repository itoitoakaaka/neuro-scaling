"""
decoder.py — 深層学習による神経デコーディング
EEGデータからのモーターイメージ分類などを行うモジュール
"""
import numpy as np


class EEGDecoder:
    """EEGデータに対するPyTorchベースのデコーダ"""

    def __init__(self, n_channels, n_times, n_classes, lr=1e-3):
        """
        Args:
            n_channels: EEGチャンネル数
            n_times: 時間ポイント数
            n_classes: 分類クラス数
            lr: 学習率
        """
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.lr = lr
        self.model = None
        self.device = None

    def build_model(self):
        """CNNモデルを構築する"""
        try:
            import torch
            import torch.nn as nn

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = nn.Sequential(
                # 時間方向の畳み込み
                nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12)),
                nn.BatchNorm2d(16),
                nn.ELU(),

                # 空間方向の畳み込み
                nn.Conv2d(16, 32, kernel_size=(self.n_channels, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.25),

                nn.Flatten(),
            ).to(self.device)

            # ダミーデータでFlatten後のサイズを計算
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times).to(self.device)
            flat_size = self.model(dummy).shape[1]

            # 分類ヘッドを追加
            self.model = nn.Sequential(
                *list(self.model.children()),
                nn.Linear(flat_size, self.n_classes),
            ).to(self.device)

            print(f"✅ モデル構築完了 (デバイス: {self.device})")
            return self.model

        except ImportError:
            print("⚠️ PyTorchがインストールされていません: pip install torch")
            return None

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """モデルを学習する

        Args:
            X_train: 学習データ (n_samples, n_channels, n_times)
            y_train: 学習ラベル (n_samples,)
            epochs: エポック数
            batch_size: バッチサイズ
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if self.model is None:
            self.build_model()

        # NumPy → Tensor変換
        X = torch.FloatTensor(X_train[:, np.newaxis, :, :]).to(self.device)
        y = torch.LongTensor(y_train).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, X_test, y_test):
        """モデルを評価する

        Args:
            X_test: テストデータ (n_samples, n_channels, n_times)
            y_test: テストラベル (n_samples,)

        Returns:
            accuracy: 精度
        """
        import torch

        self.model.eval()
        X = torch.FloatTensor(X_test[:, np.newaxis, :, :]).to(self.device)

        with torch.no_grad():
            output = self.model(X)
            preds = output.argmax(dim=1).cpu().numpy()

        accuracy = np.mean(preds == y_test)
        print(f"✅ テスト精度: {accuracy:.4f}")
        return accuracy
