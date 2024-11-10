FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip & pip install pipenv
COPY [ "Pipfile", "Pipfile.lock", "homework.py", "./" ]
RUN pipenv install --system --deploy
ENTRYPOINT [ "python", "homework.py" , "2023", "05"]
