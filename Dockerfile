FROM python:3.7-slim

COPY . /app

WORKDIR /app

# must be installed before pycocotools
RUN pip3 install Cython
RUN pip3 install -r requirements.txt
# must be installed after numpy
RUN pip3 install pycocotools

CMD ["/bin/bash"]
