@echo off
REM Скрипт для запуска теста эмбеддингов с правильными настройками для Windows
REM Этот скрипт устанавливает переменные окружения для отключения Xet Storage

echo [INFO] Настройка переменных окружения для отключения Xet Storage...

REM Отключаем Xet Storage для избежания проблем с таймаутами
set HF_HUB_DISABLE_XET=1
set HF_HUB_ENABLE_XET=0
set HF_HUB_DISABLE_XET_WARNING=1

REM Увеличиваем таймаут загрузки до 20 минут
set HF_HUB_DOWNLOAD_TIMEOUT=1200

REM Увеличиваем количество попыток
set HF_DOWNLOAD_MAX_RETRIES=5

REM Увеличиваем задержку между попытками
set HF_DOWNLOAD_RETRY_DELAY=10

echo [INFO] Переменные окружения установлены:
echo        HF_HUB_DISABLE_XET=%HF_HUB_DISABLE_XET%
echo        HF_HUB_DOWNLOAD_TIMEOUT=%HF_HUB_DOWNLOAD_TIMEOUT%
echo        HF_DOWNLOAD_MAX_RETRIES=%HF_DOWNLOAD_MAX_RETRIES%
echo.
echo [INFO] Запуск теста эмбеддингов...
echo.

REM Запускаем тест
python test_embeddings.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Тест завершился с ошибкой.
    echo [INFO] Если проблема сохраняется, попробуйте:
    echo        1. Проверить подключение к интернету
    echo        2. Увеличить таймаут: set HF_HUB_DOWNLOAD_TIMEOUT=1800
    echo        3. Загрузить модель вручную с сайта Hugging Face
    pause
    exit /b %ERRORLEVEL%
)

pause

