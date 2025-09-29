# 使い方
## 最初のセットアップ
- 好きな場所にpython 3.11.9を使用したvenv仮想環境を用意
- 好きな場所に本フォルダを配置
- venv起動用スクリプトから仮想環境を起動してから
  - `$ pip install pip-tools`
  - `$ pip-sync requirements.txt`

## ライブラリを更新したい場合
- `requirements.in`を編集
- venv起動用スクリプトから仮想環境を起動
  - `$ pip-compile requirements.in`で`requirements.txt`を生成
  - `$ pip-sync requirements.txt`