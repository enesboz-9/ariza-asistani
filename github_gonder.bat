@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

rem ===== AYARLAR =====
set GH_USER=enesboz-9
set REPO=ariza-asistani
set REMOTE=https://github.com/%GH_USER%/%REPO%.git
rem ===================

echo.
echo === 1/4 Arac kontrolu ===
where git >nul 2>nul || (echo [HATA] git kurulu degil: https://git-scm.com & goto :fail)

echo.
echo === 2/4 Guvenlik kontrolu (API anahtari dosyalarda var mi?) ===
findstr /S /I /M /R "gsk_[A-Za-z0-9]" *.py *.json *.md *.bat *.txt >nul 2>nul
if not errorlevel 1 (
  echo [HATA] Dosyalarin icinde API anahtari gibi bir metin var ^(gsk_...^). Silmeden gonderme!
  goto :fail
)

echo.
echo === 3/4 Repo hazirlaniyor ===
if not exist .git git init -b main
git add -A
git commit -m "Arac ariza asistani: guncelleme" >nul 2>nul
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %REMOTE%

rem GitHub CLI kuruluysa repo yoksa otomatik olusturmayi dene
where gh >nul 2>nul
if not errorlevel 1 (
  gh repo view %GH_USER%/%REPO% >nul 2>nul
  if errorlevel 1 (
    echo Repo bulunamadi, gh ile olusturuluyor...
    gh repo create %GH_USER%/%REPO% --public
  )
)

echo.
echo === 4/4 GitHub'a gonderiliyor ===
git push -u origin main
if errorlevel 1 (
  echo.
  echo [HATA] Push basarisiz.
  echo  - Repo yoksa https://github.com/new adresinden "%REPO%" adinda BOS bir repo olustur.
  echo  - Repo bos degilse ^(README eklendiyse^) once: git pull origin main --allow-unrelated-histories
  goto :fail
)

echo.
echo ================================================
echo  TAMAM! https://github.com/%GH_USER%/%REPO%
echo ================================================
pause
exit /b 0

:fail
echo.
echo Islem yarim kaldi. Yukaridaki hatayi kontrol et.
pause
exit /b 1
