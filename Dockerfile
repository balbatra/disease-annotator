FROM python:3.9

RUN pip install pipenv

RUN mkdir -p /usr/local/disease-bert

ENV PROJECT_DIR /usr/local/disease-bert

WORKDIR ${PROJECT_DIR}
ADD .. ${PROJECT_DIR}

RUN pipenv install --system --deploy --ignore-pipfile

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
