#!/bin/bash
# Скрипт для запуска теста эмбеддингов с правильными настройками для Linux/Mac

echo "[INFO] Настройка переменных окружения для отключения Xet Storage..."

# Отключаем Xet Storage для избежания проблем с таймаутами
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_XET=0
export HF_HUB_DISABLE_XET_WARNING=1

# Увеличиваем таймаут загрузки до 20 минут
export HF_HUB_DOWNLOAD_TIMEOUT=1200

# Увеличиваем количество попыток
export HF_DOWNLOAD_MAX_RETRIES=5

# Увеличиваем задержку между попытками
export HF_DOWNLOAD_RETRY_DELAY=10

echo "[INFO] Переменные окружения установлены:"
echo "       HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "       HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo "       HF_DOWNLOAD_MAX_RETRIES=$HF_DOWNLOAD_MAX_RETRIES"
echo ""
echo "[INFO] Запуск теста эмбеддингов..."
echo ""

# Запускаем тест
python test_embeddings.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Тест завершился с ошибкой."
    echo "[INFO] Если проблема сохраняется, попробуйте:"
    echo "       1. Проверить подключение к интернету"
    echo "       2. Увеличить таймаут: export HF_HUB_DOWNLOAD_TIMEOUT=1800"
    echo "       3. Загрузить модель вручную с сайта Hugging Face"
    exit 1
fi

