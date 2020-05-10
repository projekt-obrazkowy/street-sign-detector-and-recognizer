FROM python:3.7-slim

COPY . /app

WORKDIR /app

# must be installed before pycocotools
RUN pip3 install Cython
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
