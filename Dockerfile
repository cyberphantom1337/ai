# Используем базовый образ с Python 3.8
FROM python:3.8-slim

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем русскую локаль
RUN locale-gen ru_RU.UTF-8
ENV LANG=ru_RU.UTF-8
ENV LANGUAGE=ru_RU:en_US
ENV LC_ALL=ru_RU.UTF-8

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем необходимые файлы
COPY ./train_data.csv ./train_data.csv
COPY ./requirements.txt ./requirements.txt
COPY ./Ai.py ./Ai.py
COPY ./run.py ./run.py

# Устанавливаем зависимости
RUN pip install -r /app/requirements.txt

# Загружаем необходимые данные NLTK
RUN python -c "import nltk; nltk.download('stopwords')"

# Устанавливаем переменные окружения
ENV INPUT_PATH=/app/input
ENV OUTPUT_PATH=/app/output

# Создаем директории для входных и выходных данных
RUN mkdir -p ${INPUT_PATH} ${OUTPUT_PATH}

# Добавляем скрипт проверки входных данных
RUN echo '#!/bin/bash\n\
echo "Checking input directory..."\n\
if [ ! -d "$INPUT_PATH" ]; then\n\
    echo "Error: Input directory does not exist"\n\
    exit 1\n\
fi\n\
\n\
echo "Checking input file..."\n\
if [ ! -f "$INPUT_PATH/input.csv" ]; then\n\
    echo "Error: input.csv not found in $INPUT_PATH"\n\
    exit 1\n\
fi\n\
\n\
echo "Checking output directory..."\n\
if [ ! -d "$OUTPUT_PATH" ]; then\n\
    echo "Creating output directory..."\n\
    mkdir -p "$OUTPUT_PATH"\n\
fi\n\
\n\
echo "Starting Python application..."\n\
python /app/run.py --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Запускаем приложение через entrypoint скрипт
ENTRYPOINT ["/app/entrypoint.sh"] 