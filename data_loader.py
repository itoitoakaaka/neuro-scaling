"""
data_loader.py — EEG/MEG/ECoGデータの読み込みと前処理
"""
import numpy as np


def load_eeg_from_mne(subject_id, runs, dataset="eegbci"):
    """MNEのサンプルデータセットからEEGデータを読み込む

    Args:
        subject_id: 被験者ID
        runs: 試行番号のリスト
        dataset: データセット名 ("eegbci" など)

    Returns:
        raw: MNE Rawオブジェクト
    """
    import mne
    from mne.datasets import eegbci

    if dataset == "eegbci":
        fnames = eegbci.load_data(subject_id, runs)
        raws = [mne.io.read_raw_edf(f, preload=True) for f in fnames]
        raw = mne.concatenate_raws(raws)
        # 標準モンタージュを設定
        mne.datasets.eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return raw
    else:
        raise ValueError(f"未対応のデータセット: {dataset}")


def preprocess(raw, l_freq=1.0, h_freq=40.0):
    """バンドパスフィルタリングと前処理

    Args:
        raw: MNE Rawオブジェクト
        l_freq: ローパス周波数
        h_freq: ハイパス周波数

    Returns:
        raw: フィルタリング済みRawオブジェクト
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def extract_epochs(raw, event_id=None, tmin=-0.5, tmax=1.0):
    """エポック抽出

    Args:
        raw: MNE Rawオブジェクト
        event_id: イベントIDの辞書
        tmin: エポック開始時刻（秒）
        tmax: エポック終了時刻（秒）

    Returns:
        epochs: MNE Epochsオブジェクト
    """
    import mne

    events, event_dict = mne.events_from_annotations(raw)
    if event_id is None:
        event_id = event_dict

    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, 0), preload=True)
    return epochs


def epochs_to_array(epochs):
    """エポックをNumPy配列に変換

    Args:
        epochs: MNE Epochsオブジェクト

    Returns:
        X: データ配列 (n_epochs, n_channels, n_times)
        y: ラベル配列 (n_epochs,)
    """
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y
