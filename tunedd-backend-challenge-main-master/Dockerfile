FROM python:3.10-slim

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR .

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install project dependencies using Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
# Ignore virtual environment
RUN poetry config virtualenvs.create false
# Install Poetry dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the project
COPY . .

# Run the application
CMD ["poetry", "run", "uvicorn", "tunedd_api.main1:app", "--host", "0.0.0.0", "--port", "8000"]


