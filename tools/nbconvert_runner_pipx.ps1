# nbconvert_runner.ps1  (PowerShell 7+)
# 番号選択で:
# 1) 親フォルダに notebooks / notebook を探索
# 2) 変換形式 (1: html / 2: script / 3: cleared notebook)
# 3) 対象 (1: 全部 / 2: 1ファイル) → 1ファイルは番号選択
# 4) 実行（常に jupyter-nbconvert コマンドを使用）

[CmdletBinding()]
param()

try { Set-Variable -Name PSNativeCommandUseErrorActionPreference -Value $true -Scope Global -ErrorAction SilentlyContinue } catch {}
$ErrorActionPreference = "Stop"

function Fail([string]$Message, [int]$Code = 1) {
  Write-Host "[ERROR] $Message" -ForegroundColor Red
  try { $null = Read-Host "Enter を押すと閉じます" } catch {}
  exit $Code
}

#--- 1)グローバル jupyter-nbconvert の解決（PATH 未設定なら即 Fail）---
function Resolve-JupyterNbconvert {
  try {
    $cmd = Get-Command "jupyter-nbconvert" -ErrorAction Stop
    return $cmd.Source
  } catch {
    Fail "jupyter-nbconvert が見つかりません。pipx 推奨: `pipx install jupyter` → `pipx ensurepath` を実行し、新しいターミナルで再試行してください。"
  }
}

# --- nbconvert コマンドの特定 ---
$nbconvertCmd = Resolve-JupyterNbconvert
# バージョン表示
$nv = & $nbconvertCmd --version
Write-Host "[STEP] 使用する nbconvert: $nbconvertCmd  (ver $nv)" -ForegroundColor Yellow


# --- 2) 親階層で notebooks / notebook を探索 ---
$parent = Split-Path -Parent $PSScriptRoot
$cands  = @("notebooks","notebook")
$nbDir  = $null
foreach ($name in $cands) {
  $p = Join-Path $parent $name
  if (Test-Path $p -PathType Container) { $nbDir = $p; break }
}
if (-not $nbDir) { Fail "親フォルダ '$parent' に 'notebooks' もしくは 'notebook' フォルダが見つかりません。" }
Write-Host "[STEP] Notebook ルート: $nbDir" -ForegroundColor Yellow

# --- 3) 変換形式（番号選択）---
Write-Host "変換形式を選択してください。" -ForegroundColor Cyan
Write-Host "[1] html"
Write-Host "[2] script (.py)"
Write-Host "[3] cleared notebook (出力削除済みの .ipynb)"
$sel = Read-Host "番号を入力してください (1/2/3)"

$to = $null
$outName = $null
$useClear = $false

switch ($sel) {
  "1" { $to = "html";     $outName = "html"    }
  "2" { $to = "script";   $outName = "py"      }
  "3" { $to = "notebook"; $outName = "nb_cleared"; $useClear = $true }
  default { Fail "不正な入力です。1 / 2 / 3 から選んでください。" }
}

$outDir = Join-Path $nbDir $outName
if (-not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Path $outDir | Out-Null
}
Write-Host "[STEP] 出力先: $outDir" -ForegroundColor Yellow

# --- 4) 対象 .ipynb を列挙 ---
$allIpynb = Get-ChildItem -Path $nbDir -Filter *.ipynb |
            Where-Object { $_.FullName -notmatch '\\\.ipynb_checkpoints\\' }

if (-not $allIpynb) { Fail "変換対象の .ipynb が見つかりませんでした。" }

# --- 5) 対象（1ファイル or 全部）番号選択 ---
Write-Host "変換対象を選択してください。" -ForegroundColor Cyan
Write-Host "[1] 1ファイルだけ"
Write-Host "[2] 全部"
$mode = Read-Host "番号を入力してください (1/2)"

$targets = $null

if ($mode -eq "1") {
  # 相対パス一覧で番号選択
  $rel = $allIpynb | ForEach-Object { $_.FullName.Substring($nbDir.Length + 1) }
  Write-Host "どの notebook を変換しますか？（番号で選択）" -ForegroundColor Cyan
  for ($i=0; $i -lt $rel.Count; $i++) {
    "{0,3}. {1}" -f ($i+1), $rel[$i] | Write-Host
  }
  $choice = Read-Host ("1～{0} の番号を入力" -f $rel.Count)
  if (-not ($choice -as [int])) { Fail "数字を入力してください。" }
  $idx = [int]$choice
  if ($idx -lt 1 -or $idx -gt $rel.Count) { Fail "範囲外の番号が選択されました。" }
  $targets = @($allIpynb[$idx-1])
}
elseif ($mode -eq "2") {
  $targets = $allIpynb
}
else {
  Fail "不正な入力です。1 または 2 を選んでください。"
}

# --- 6) nbconvert 実行（jupyter-nbconvert 固定） ---
$ok = 0; $ng = 0
foreach ($nb in $targets) {
  Write-Host ("-> 変換中: {0}" -f $nb.FullName)
  try {
    if ($useClear) {
      $args = @(
        "--to","notebook",
        "--ClearOutputPreprocessor.enabled=True",
        "--output-dir",$outDir,
        "$($nb.FullName)"
      )
    } else {
      $args = @(
        "--to",$to,
        "--output-dir",$outDir,
        "$($nb.FullName)"
      )
    }

    & $nbconvertCmd @args
    $ok++
  } catch {
    Write-Host ("   失敗: {0}" -f $nb.FullName) -ForegroundColor Red
    $ng++
  }
}

Write-Host "=== 完了 === 成功:$ok 失敗:$ng 出力:$outDir" -ForegroundColor Cyan
try { $null = Read-Host "Enter を押すと閉じます" } catch {}
