FROM python:3.8.1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY ./stable ./stable
COPY *.py ./
EXPOSE 80
ENV NAME World
CMD ["python","Main.py"]
