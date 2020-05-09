# street-sign-detector-and-recognizer

## Usage

```sh
./app.py *.jpg
```

## Development

### Install `tensorflow`

```sh
pip3 install --ignore-installed --upgrade tensorflow==1.14
```

### Install versioned dependencies

```sh
pip3 install numpy==1.16.4 pillow==6.2.1 lxml==4.4.1 jupyter==1.0.0 matplotlib==3.1.1 opencv-python==3.4.2.17 pathlib==1.0.1
```

### Install dependencies

```sh
pip3 install pycocotools pandas
```

### Install `labelImg`

```sh
pip3 install labelImg 
```

### Download images

Fetch images of signs to `images/` directory.

### Label images

Launch `labelImg` program and choose `images/` directory for both input and output.
Correctly label images with proper names.

### Partition images

```sh
./scripts/partition_images.py
```

### Convert `.xml` files to `.csv`

```sh
./scripts/xml_to_csv.py
```

### Convert `.csv` files to TensorFlow's `.record`

```sh
./scripts/csv_to_tfrecord.py
```

### Train model

```sh
./scripts/train.sh
```

### Export inference graph

```sh
./scripts/export_inference_graph.sh
```
