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

Генерация кода:
```
 python <path to inference.py> <api key> <your equation name> <output script name>
``` 