#!/bin/bash
cd "$(dirname "$0")"
xhost +local:root > /dev/null 2>&1
docker compose up