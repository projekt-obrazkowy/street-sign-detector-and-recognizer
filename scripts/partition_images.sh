#!/usr/bin/env bash

set -eu

shopt -s nullglob

IMAGES_DIR="images"
IMAGES_TRAIN_DIR="$IMAGES_DIR/train"
IMAGES_TEST_DIR="$IMAGES_DIR/test"

rm -rf "$IMAGES_TEST_DIR"
mkdir "$IMAGES_TEST_DIR"
rm -rf "$IMAGES_TRAIN_DIR"
mkdir "$IMAGES_TRAIN_DIR"

for file in $IMAGES_DIR/*.jpg $IMAGES_DIR/*.xml; do
    ln -s "$IMAGES_DIR/$file" "$IMAGES_TRAIN_DIR"
done


for file in $(ls -1 "$IMAGES_TRAIN_DIR" | head -n 10); do
    ln -s "$IMAGES_TRAIN_DIR/$file" "$IMAGES_TEST_DIR"
done
