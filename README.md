# LFORD

## 概要
LFORD ( **L**LM-based **For**ced **D**ecoding )は、LLM の文生成確率に基づく教師なし品質推定手法です。
![method](method.png)

LFORDは、様々な言語生成タスクに適用可能ですが、このレポジトリでは英日の機械翻訳向けに実装しています。

## 環境設定

#### 【初回のみ】リポジトリのクローンと仮想環境の作成
```
git clone https://github.com/nara-wu-slc/LFORD_tutorial.git
cd LFORD_tutorial
python3 -m venv --prompt . .venv
```

#### 【毎回必須】仮想環境の有効化
```
source .venv/bin/activate
```

#### 【初回のみ】必要なライブラリのインストール
```
pip3 install torch==2.6.0 typed-argument-parser==1.10.1 tqdm==4.67.1 transformers==4.49.0 accelerate==1.4.0
```

#### 【毎回必須】モデル保存用のキャッシュディレクトリの設定を読み込む
```
source /slc/share/dot.zshrc.slc
```

## 使用方法
#### サンプル
```
python evaluate.py
    --src_file_path sample/sample.en
    --tgt_file_path sample/sample.ja
    --output_file_dir sample/result
```

## 文献情報
樽本空宙, 梶原智之, 二宮崇. 大規模言語モデルの文生成確率を用いた教師なし品質推定.<br>
言語処理学会第31回年次大会  [[pdf](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/P7-10.pdf)]
