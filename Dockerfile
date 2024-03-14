
FROM python:3.12
WORKDIR /service
COPY requirements.txt .
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" , 'app.py' ]