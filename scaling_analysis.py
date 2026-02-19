"""
scaling_analysis.py — スケーリング則の解析
データサイズと性能の関係（スケーリング則）を調べるモジュール
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_scaling_curve(X, y, data_fractions=None, n_cv=5):
    """データサイズに対するモデル性能のスケーリング曲線を計算する

    Args:
        X: 特徴量配列 (n_samples, n_features)
        y: ラベル配列 (n_samples,)
        data_fractions: データの割合リスト (例: [0.1, 0.2, ..., 1.0])
        n_cv: 交差検証の分割数

    Returns:
        fractions: データの割合
        scores_mean: 各割合での平均精度
        scores_std: 各割合での標準偏差
    """
    if data_fractions is None:
        data_fractions = np.arange(0.1, 1.1, 0.1)

    n_total = len(X)
    scores_mean = []
    scores_std = []

    scaler = StandardScaler()

    for frac in data_fractions:
        n_subset = max(int(n_total * frac), n_cv + 1)
        indices = np.random.choice(n_total, n_subset, replace=False)
        X_sub = scaler.fit_transform(X[indices])
        y_sub = y[indices]

        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_sub, y_sub, cv=min(n_cv, n_subset))
        scores_mean.append(np.mean(scores))
        scores_std.append(np.std(scores))

    return data_fractions, np.array(scores_mean), np.array(scores_std)


def fit_power_law(fractions, scores):
    """スケーリング曲線にべき乗則をフィットする

    y = a * x^b + c

    Args:
        fractions: データの割合
        scores: 各割合での精度

    Returns:
        params: フィットパラメータ (a, b, c)
    """
    from scipy.optimize import curve_fit

    def power_law(x, a, b, c):
        return a * np.power(x, b) + c

    try:
        params, _ = curve_fit(power_law, fractions, scores, p0=[1.0, 0.5, 0.5])
        return params
    except RuntimeError:
        print("⚠️ べき乗則のフィットに失敗しました")
        return None


def plot_scaling_curve(fractions, scores_mean, scores_std, params=None,
                       output_path="scaling_curve.png"):
    """スケーリング曲線をプロットする

    Args:
        fractions: データの割合
        scores_mean: 平均精度
        scores_std: 標準偏差
        params: べき乗則のフィットパラメータ
        output_path: 出力画像パス
    """
    plt.figure(figsize=(8, 5))
    plt.errorbar(fractions, scores_mean, yerr=scores_std,
                 fmt='o-', capsize=5, label='実測値')

    if params is not None:
        a, b, c = params
        x_fit = np.linspace(fractions[0], fractions[-1], 100)
        y_fit = a * np.power(x_fit, b) + c
        plt.plot(x_fit, y_fit, 'r--',
                 label=f'べき乗則: y={a:.3f}·x^{b:.3f}+{c:.3f}')

    plt.xlabel('データ割合')
    plt.ylabel('精度 (Accuracy)')
    plt.title('Neural Scaling Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ スケーリング曲線を保存: {output_path}")
