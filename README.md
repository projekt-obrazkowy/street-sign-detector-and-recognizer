# street-sign-detector

## Usage

```sh
./app.py *.jpg
```

## Development

### Install requirements

```sh
pip3 install -r requirements.txt
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

### Generate annotations

```sh
./pre-train.py
```

### Train model

```sh
./train.py
```

### Export inference graph

```sh
./post-train.py
```
