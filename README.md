# Kaggle House Prices 住宅価格予測

[![notebooks](https://github.com/T0bbuT/kaggle_house_price/actions/workflows/notebooks.yml/badge.svg)](https://github.com/T0bbuT/kaggle_house_price/actions/workflows/notebooks.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![uv](https://img.shields.io/badge/env-uv-purple)

Kaggle のコンペ [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) に取り組んだ分析です。
米国アイオワ州エイムズの住宅 1,460 件・79 個の特徴量から販売価格（`SalePrice`）を予測します。

## 結果

5-fold クロスバリデーションでの評価です（評価指標はコンペと同じ **対数価格の RMSE**、補助的に MAPE）。

| モデル | CV RMSE（対数価格） | CV MAPE | ノートブック |
|---|---|---|---|
| **Lasso**（Optuna でチューニング） | **0.1112** | **7.97%** | [02](notebooks/02_tuned_models.ipynb) |
| Ridge（Optuna でチューニング） | 0.1122 | 8.00% | [02](notebooks/02_tuned_models.ipynb) |
| LightGBM（Optuna でチューニング） | 0.1189 | 8.30% | [02](notebooks/02_tuned_models.ipynb) |
| Ridge（グリッドサーチ、ベースライン）※ | 0.1138 | – | [01](notebooks/01_baseline.ipynb) |

※ 01 は前処理が異なり、学習データの 80% に対する 5-fold CV の値です。

- 予測価格は平均して実際の価格から約 8% ずれる水準です。
- 目的変数・特徴量の分布を整えたことで、線形モデル（Lasso / Ridge）が勾配ブースティング（LightGBM）をわずかに上回りました。
- 乱数シードを固定しているため、同じ環境で実行すれば同じ数値が再現されます。

<p>
  <img src="docs/images/saleprice_before.png" width="49%" alt="変換前の SalePrice の分布">
  <img src="docs/images/saleprice_after_boxcox.png" width="49%" alt="Box-Cox 変換後の SalePrice の分布">
</p>

左: 元の `SalePrice` は右に大きく歪んでいる（歪度 1.88）。右: Box-Cox 変換後はほぼ正規分布に近づいた。

## 分析の流れ

| ノートブック | 内容 |
|---|---|
| [01_baseline.ipynb](notebooks/01_baseline.ipynb) | 教材に沿った初期版。EDA → 欠損値補完 → 外れ値除去 → 対数変換・Box-Cox 変換 → 7 種類のモデル比較 → Ridge のグリッドサーチ |
| [02_tuned_models.ipynb](notebooks/02_tuned_models.ipynb) | 01 を踏まえた改良版。前処理を整理し直し、特徴量を追加して LightGBM / Ridge / Lasso を Optuna でチューニング |

02 の主な処理:

1. **EDA**: 目的変数の分布、相関係数、相関の高い特徴量との散布図（plotly）
2. **外れ値除去**: 居住面積（`GrLivArea`）が大きいのに価格が極端に低い 2 件を除外
3. **欠損値補完**: 「設備がない」ことを表す欠損は `"None"` / `0`、`LotFrontage` は地域（`Neighborhood`）ごとの中央値、その他は最頻値。ほぼ単一値の列（`PoolQC`, `Utilities`, `PoolArea`）は削除
4. **特徴量エンジニアリング**: 総床面積 `TotalSF`、総浴室数 `TotalBathrooms`、ポーチ総面積 `TotalPorchSF`、2 階・ガレージ・地下室・暖炉の有無フラグなど
5. **エンコーディング**: 品質・状態など順序のあるカテゴリは Ordinal Encoding、それ以外は One-hot Encoding
6. **分布の調整**: 目的変数は Box-Cox 変換、歪度 0.75 超の特徴量は Yeo-Johnson 変換
7. **モデリング**: 5-fold CV を目的関数にして Optuna（TPE, 各 50 試行）でハイパーパラメータ探索し、最良パラメータで提出用 CSV を作成

## 設計の判断

なぜそうしたのかをまとめます。該当箇所は [02_tuned_models.ipynb](notebooks/02_tuned_models.ipynb) です。

- **評価はコンペの指標に揃えた**: 目的変数は Box-Cox 変換して学習するが、変換後の値のまま誤差を測ると、コンペの指標とは別のものを最適化してしまう。そこで交差検証では予測を元の価格に逆変換し、コンペと同じ「対数価格の RMSE」で評価して、その値を Optuna の目的関数にした。
- **線形モデルが力を出せる前処理を用意した**: 最初は前処理をほとんどせずに LightGBM で試し、次に線形モデルを試すために、目的変数・特徴量の分布の調整（Box-Cox / Yeo-Johnson 変換）と、順序を考慮したカテゴリのエンコーディングを準備した。結果として Lasso が LightGBM を上回った。
- **`NA` の意味を列ごとに確かめた**: このデータでは `NA` が「欠損」ではなく「その設備がない」を表す列が多い。特徴量の説明を日本語の表（`docs/data_description_jp.csv`）にして欠損数の表と突き合わせ、列ごとに `"None"` / `0` / 最頻値のどれで補うかを決めた。
- **`LotFrontage` の補完は、何でグループ分けするかを比べて決めた**: 欠損の多い `LotFrontage`（道路に面した間口の長さ）について、すべてのカテゴリ変数との箱ひげ図を描いて比べ、差がはっきり出た `Neighborhood`（地域）ごとの中央値で補完した。
- **順序のあるカテゴリは取りこぼさずに数値化**: 各カテゴリ列の値の組み合わせを自動で集計し、同じ組み合わせを持つ列をまとめてから（[メモ](docs/notes/OrdinalEncoding準備.txt)）、`None < Po < Fa < TA < Gd < Ex` のような順序を列ごとに定義した。
- **外れ値は行を特定してから除いた**: plotly の散布図で行番号を確認し、居住面積が大きいのに価格が極端に低い 2 件だけを除外した。
- **補足**: 欠損補完の中央値と Yeo-Johnson 変換は、コンペの慣例に従い訓練データとテストデータを合わせた全体で計算している。実務で使う場合は、訓練データだけで計算してテストデータに当てはめる。

## 環境構築と実行

パッケージ管理には [uv](https://docs.astral.sh/uv/) を使っています。依存パッケージのバージョンは `uv.lock` で固定しているので、誰の環境でも同じ構成が再現できます。Python 3.11 も uv が自動で用意します。

### 1. uv をインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. リポジトリを取得して環境を作る

```bash
git clone https://github.com/T0bbuT/kaggle_house_price.git
cd kaggle_house_price
uv sync
```

`.venv/` に仮想環境が作られます。

### 3. データをダウンロード

コンペのデータは Kaggle の規約に従いリポジトリには含めていないため、[Kaggle CLI](https://github.com/Kaggle/kaggle-api)（`uv sync` で一緒に入ります）で取得します。

1. Kaggle にログインし、[コンペのページ](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules) でルールに同意する
2. [Kaggle の Settings](https://www.kaggle.com/settings) の「API」から API トークンを作成し、表示された手順どおりに保存する（`~/.kaggle/` に置くか、環境変数に設定）
3. 次のコマンドでダウンロードする

```bash
uv run kaggle competitions download -c house-prices-advanced-regression-techniques -p data/input --unzip
```

`data/input/` に `train.csv`, `test.csv`, `sample_submission.csv`, 公式の特徴量説明 `data_description.txt` が展開されます（`data/` は Git 管理外）。

ブラウザで [Data タブ](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) から zip をダウンロードし、`data/input/` に展開しても構いません。

### 4. ノートブックを実行

```bash
# JupyterLab で開く
uv run jupyter lab

# または、コマンドラインで最初から最後まで実行する（02 は数分かかります）
uv run jupyter nbconvert --to notebook --execute notebooks/02_tuned_models.ipynb --output-dir outputs/executed
```

VS Code を使う場合は、ノートブックのカーネルに `.venv` の Python を選択してください。

提出用 CSV は `outputs/submissions/` に出力されます（`outputs/` は Git 管理外）。

GitHub Actions でも push / PR ごとに Kaggle からデータを取得して両方のノートブックを最初から実行し、壊れていないことを確認しています（リポジトリの Secrets に Kaggle の API トークン `KAGGLE_API_TOKEN`、または `KAGGLE_USERNAME` と `KAGGLE_KEY` が必要です）。

## ディレクトリ構成

```
.
├── data/
│   └── input/               # コンペ配布データの置き場所（Git 管理外、手順 3 でダウンロード）
├── docs/
│   ├── data_description_jp.csv  # 特徴量説明の日本語訳（ノートブックから参照）
│   ├── images/                  # README 用の図
│   └── notes/                   # 分析中の作業メモ
├── notebooks/
│   ├── 01_baseline.ipynb
│   └── 02_tuned_models.ipynb
├── outputs/                 # ノートブックの出力先（Git 管理外）
├── pyproject.toml           # 依存パッケージの定義
└── uv.lock                  # 依存パッケージのバージョン固定
```

ノートブックは出力をクリアした状態でコミットしています。

## 振り返り

**工夫した点**

- 特徴量が多く英語なので、日本語訳の一覧表（`docs/data_description_jp.csv`）を作り、欠損値の集計表と突き合わせながら補完方法を決めた
- データ中の `NA` が「欠損」ではなく「設備がない」を意味する列が多く、列ごとに意味を確認して `"None"` / `0` で補完した
- 外れ値の特定には plotly のインタラクティブな散布図を使い、該当する行を直接確認した

**課題と今後の改善案**

- `GarageYrBlt`（ガレージ建築年）の欠損をガレージなしとして 0 で補完したが、より適切な扱いがありそう
- `LotFrontage` の補完を、専用の予測モデルで行う方法も試したい
- ドメイン知識に基づく特徴量エンジニアリングや、複数モデルのアンサンブルで精度向上の余地がある

詳しい作業メモは [docs/notes/](docs/notes/) にあります。
