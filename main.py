"""
main.py — neuro-scaling エントリーポイント

使い方:
    python main.py download          # サンプルデータのダウンロード
    python main.py scaling           # スケーリング則の解析
    python main.py decode            # デコーディングモデルの学習
    python main.py all               # 全パイプライン実行
"""
import sys
import numpy as np
from data_loader import load_eeg_from_mne, preprocess, extract_epochs, epochs_to_array
from scaling_analysis import compute_scaling_curve, fit_power_law, plot_scaling_curve
from decoder import EEGDecoder


def download_data():
    """サンプルEEGデータをダウンロードする"""
    print("📥 サンプルデータをダウンロード中...")
    # Motor imagery: 左手 vs 右手 (runs 4, 8, 12)
    raw = load_eeg_from_mne(subject_id=1, runs=[4, 8, 12])
    print(f"✅ ダウンロード完了: {len(raw.ch_names)} チャンネル, {raw.n_times} サンプル")
    return raw


def run_scaling_analysis():
    """スケーリング則の解析を実行する"""
    print("\n📊 スケーリング則の解析を開始...")

    # 複数被験者のデータを読み込み
    all_X, all_y = [], []
    for subj in range(1, 6):  # 被験者1-5
        print(f"  被験者 {subj} のデータを処理中...")
        raw = load_eeg_from_mne(subject_id=subj, runs=[4, 8, 12])
        raw = preprocess(raw)
        epochs = extract_epochs(raw)
        X, y = epochs_to_array(epochs)
        # 特徴量: チャンネルごとの平均パワー
        X_feat = np.mean(X ** 2, axis=-1)  # (n_epochs, n_channels)
        all_X.append(X_feat)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.hstack(all_y)

    # スケーリング曲線を計算
    fractions, scores_mean, scores_std = compute_scaling_curve(X_all, y_all)

    # べき乗則フィット
    params = fit_power_law(fractions, scores_mean)

    # プロット
    plot_scaling_curve(fractions, scores_mean, scores_std, params)


def run_decoding():
    """デコーディングモデルの学習を実行する"""
    print("\n🧠 神経デコーディングを開始...")

    # データ読み込み
    raw = load_eeg_from_mne(subject_id=1, runs=[4, 8, 12])
    raw = preprocess(raw)
    epochs = extract_epochs(raw)
    X, y = epochs_to_array(epochs)

    # ラベルを0始まりに変換
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])

    # データ分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42
    )

    # デコーダの構築と学習
    n_channels, n_times = X.shape[1], X.shape[2]
    n_classes = len(unique_labels)

    decoder = EEGDecoder(n_channels, n_times, n_classes)
    decoder.build_model()
    decoder.train(X_train, y_train, epochs=30)
    decoder.evaluate(X_test, y_test)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "download":
        download_data()
    elif command == "scaling":
        run_scaling_analysis()
    elif command == "decode":
        run_decoding()
    elif command == "all":
        download_data()
        run_scaling_analysis()
        run_decoding()
    else:
        print(f"❌ 不明なコマンド: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
