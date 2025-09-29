# 起動.ps1

# このスクリプトのある場所に移動
Set-Location -Path $PSScriptRoot

# 実行したいスクリプトを呼び出し
pwsh -NoExit -ExecutionPolicy Bypass -File "C:\Users\shang\venvs\venv_kaggle_house_price\Scripts\Activate.ps1"