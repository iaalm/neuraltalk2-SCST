#!/bin/bash

cd coco-caption
flock lock python2 myeval.py $2 $1
cd ../
