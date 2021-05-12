#!/usr/bin/env bash

DIR=./examples/data
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/coupland1.jpg -o ${DIR}/results/teapot_coupland1.gif -rd ${DIR}/results
python ./examples/run.py -im ${DIR}/meshes/bunny.obj -is ${DIR}/styles/gogh2.jpg -o ${DIR}/results/bunny_gogh2.gif -rd ${DIR}/results
# python ./examples/run.py -im ${DIR}/meshes/bunny.obj -is ${DIR}/styles/klee1.jpg -o ${DIR}/results/bunny_klee1.gif -rd ${DIR}/results
# python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/lettl1.jpg -o ${DIR}/results/teapot_lettl1.gif -rd ${DIR}/results
# python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/sketch1.jpg -o ${DIR}/results/teapot_sketch1.gif -rd ${DIR}/results
# python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/monet4.jpg -o ${DIR}/results/teapot_monet4.gif -rd ${DIR}/results
