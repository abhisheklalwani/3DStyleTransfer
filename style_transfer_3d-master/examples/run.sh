#!/usr/bin/env bash

DIR=./examples/data
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/coupland1.jpg -o ${DIR}/results/teapot_coupland1.gif -rd ${DIR}/results
python ./examples/run.py -im ${DIR}/meshes/bunny.obj -is ${DIR}/styles/gris1.jpg -o ${DIR}/results/bunny_gris1.gif -rd ${DIR}/results
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/gris1.jpg -o ${DIR}/results/teapot_gris1.gif -rd ${DIR}/results
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/gogh2.jpg -o ${DIR}/results/teapot_gogh2.gif -rd ${DIR}/results
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/picasso.jpg -o ${DIR}/results/teapot_picasso.gif -rd ${DIR}/results
