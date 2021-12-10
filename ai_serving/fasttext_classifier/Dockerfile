FROM python:3.7

WORKDIR /usr/lib
RUN git clone "https://github.com/facebookresearch/fastText.git" && cd fastText && pip install .

WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["echo", "Ready"]
