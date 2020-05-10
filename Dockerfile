FROM python:3.7-slim

COPY . /app

WORKDIR /app

# gcc is needed for pycocotools
RUN apt-get update && apt-get install -yq build-essential

# must be installed before pycocotools
RUN pip3 install Cython
RUN pip3 install -r requirements.txt
# must be installed after numpy
RUN pip3 install pycocotools

CMD ["/bin/bash"]
