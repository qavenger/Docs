# 扫描 各档位/src 下的 png，生成 src/files.js（页面据此自动列出图片）
# 同时按「有 src 的文件夹」重建根目录 tiers.js —— 顶部 Tab 数量/名称 = 文件夹数量/名称
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }

$enc = New-Object System.Text.UTF8Encoding($false)

$dirs = Get-ChildItem -LiteralPath $root -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'src') } |
        Sort-Object Name

if (-not $dirs) {
    Write-Host '没找到档位文件夹（需要有 <档位名>\src 这样的结构）'
    exit
}

# ---- 1) 每个档位：补 config.js（缺才补，放在档位目录下）+ 生成 src/files.js
foreach ($d in $dirs) {
    $src  = Join-Path $d.FullName 'src'
    $pngs = Get-ChildItem -LiteralPath $src -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -ieq '.png' } |
            Sort-Object Name

    $names = @($pngs | ForEach-Object { $_.Name })

    $cfgPath = Join-Path $d.FullName 'config.js'
    if (-not (Test-Path -LiteralPath $cfgPath)) {
        $left  = if ($names.Count -ge 1) { $names[0] } else { '' }
        $right = if ($names.Count -ge 2) { $names[1] } else { $left }
        $stub = ("// {0} · 扫描脚本自动创建的占位配置，按需改机型/槽位`r`n" -f $d.Name) +
                "window.GSR_CONFIG = {`r`n" +
                ("  tier: `"{0}`",`r`n" -f $d.Name) +
                "  device: `"`",`r`n  hardware: `"`",`r`n  accent: `"#f0a24a`",`r`n  mode: `"wipe`",`r`n" +
                ("  left: `"{0}`",`r`n  right: `"{1}`",`r`n" -f $left, $right) +
                "  shots: []`r`n};`r`n"
        [System.IO.File]::WriteAllText($cfgPath, $stub, $enc)
        Write-Host ("{0}: 新建占位 config.js（{0}/config.js，记得补机型和左右默认图）" -f $d.Name)
    }

    $body = ''
    if ($names.Count -gt 0) {
        $body = ($names | ForEach-Object { '  "' + $_ + '"' }) -join ",`r`n"
    }

    $text = "// 自动生成，勿手改：扫描 $($d.Name)/src 下的 png。重新扫描请运行 扫描图片.bat`r`n" +
            "window.GSR_FILES = [`r`n$body`r`n];`r`n"

    [System.IO.File]::WriteAllText((Join-Path $src 'files.js'), $text, $enc)
    Write-Host ("{0}: {1} 张图" -f $d.Name, $names.Count)
}

# ---- 2) 根目录 tiers.js：Tab 名 = 文件夹名（沿用上次顺序，新文件夹追加到末尾）
$tiersPath = Join-Path $root 'tiers.js'
$oldOrder = @()
if (Test-Path -LiteralPath $tiersPath) {
    $oldText = Get-Content -LiteralPath $tiersPath -Raw -Encoding UTF8
    $opt = [System.Text.RegularExpressions.RegexOptions]::Singleline
    $m = [regex]::Match($oldText, '\[(.*?)\]', $opt)
    if ($m.Success) {
        foreach ($mm in [regex]::Matches($m.Groups[1].Value, '"([^"]+)"')) {
            $oldOrder += $mm.Groups[1].Value
        }
    }
}

$found = @($dirs | ForEach-Object { $_.Name })
$ordered = @()
foreach ($n in $oldOrder) { if (($found -contains $n) -and ($ordered -notcontains $n)) { $ordered += $n } }
foreach ($n in $found)   { if ($ordered -notcontains $n) { $ordered += $n } }

$list = ($ordered | ForEach-Object { '"' + $_ + '"' }) -join ', '
$tiersText = ("// 自动生成，勿手改：由 扫描图片.bat 按「有 src 的文件夹」生成，Tab 名 = 文件夹名`r`n" +
              "// 顺序沿用上一次的内容，新文件夹追加在末尾；想调顺序直接改下面这行，下次扫描会保留。`r`n" +
              "window.GSR_TIERS = [$list];`r`n")

[System.IO.File]::WriteAllText($tiersPath, $tiersText, $enc)
Write-Host ("Tab: {0}" -f ($ordered -join ' / '))
