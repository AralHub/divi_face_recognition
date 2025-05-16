# Используем базовый образ с Python 3.11
FROM python:3.12

# Устанавливаем Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Добавляем папку проекта и устанавливаем рабочую директорию
WORKDIR /app

# Добавляем Poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Копируем файлы конфигурации Poetry (pyproject.toml и poetry.lock) в контейнер
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости с помощью Poetry
RUN poetry install --no-root

# Копируем остальной код в контейнер
COPY . .

# Запускаем приложение с помощью Gunicorn и Uvicorn
CMD ["poetry", "run", "gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
