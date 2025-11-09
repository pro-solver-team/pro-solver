Клонировать репозиторий:
```
git clone https://github.com/arunashamil/pro-solver.git
cd pro-solver
```
Управление зависимостями:
```
pip install poetry==2.2.1
poetry install
poetry env activate
```
Создать векторную базу данных Chroma:
```
cd pro_solver/database/
poetry run python create_database.py
```

Хард-код инференс:
```
cd ../infer/
poetry run python infer.py API_KEY YOUR_QUESTION OUTPUT_SOLVER_NAME
``` 