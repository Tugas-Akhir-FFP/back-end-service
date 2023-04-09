FROM python:latest

COPY . /app

WORKDIR /app

RUN pip install virtualenv

RUN virtualenv venv

RUN . venv/bin/activate && pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
