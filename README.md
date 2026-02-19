# neuro-scaling

大規模脳データ（EEG / MEG / ECoG）を用いて スケーリング則の解析 と 深層学習による神経デコーディング を行うリポジトリ

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

```bash
# データのダウンロード
python main.py download --dataset eeg_motor

# スケーリング則の解析
python main.py scaling --data_dir data/

# デコーディングモデルの学習
python main.py decode --data_dir data/ --epochs 50
```

## プロジェクト構成

```
neuro-scaling/
├── main.py                # エントリーポイント
├── scaling_analysis.py    # スケーリング則の解析
├── decoder.py             # 深層学習デコーダ
├── data_loader.py         # データ読み込み・前処理
├── requirements.txt       # 依存ライブラリ
└── README.md
```
