#!/bin/bash
cd "$(dirname "$0")"
xhost +local:root > /dev/null 2>&1
if [ -e /dev/ttyUSB0 ]; then
    export GRBL_DEVICE=/dev/ttyUSB0
    echo "GRBL станок найден ($GRBL_DEVICE). Запуск с поддержкой перемещения."
else
    export GRBL_DEVICE=/dev/null
    echo "GRBL станок НЕ найден. Запуск без режима перемещения."
fi
docker compose up