@echo off
REM Скрипт для запуска полного RAG пайплайна с правильными настройками

echo [INFO] Настройка переменных окружения для отключения Xet Storage...

REM Отключаем Xet Storage
set HF_HUB_DISABLE_XET=1
set HF_HUB_ENABLE_XET=0
set HF_HUB_DISABLE_XET_WARNING=1

REM Отключаем ONNX оптимизацию
set SENTENCE_TRANSFORMERS_DISABLE_ONNX=1
set ST_DISABLE_ONNX=1

REM Увеличиваем таймаут загрузки
set HF_HUB_DOWNLOAD_TIMEOUT=1200

echo [INFO] Запуск полного RAG пайплайна...
echo.

REM Запускаем пайплайн
python scripts/rag_pipeline_full.py --input data/processed/rag_data.jsonl --output data/output/rag_processed.json --embedding-provider sentence_transformers

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Пайплайн завершился с ошибкой.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Пайплайн завершен успешно!
pause

