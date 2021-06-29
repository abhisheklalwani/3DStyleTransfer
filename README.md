# 3D StyleTransfer

This repository contains implementation of the Final Project of COMPSCI 674 course at UMass Amherst. <br>
Contributors: <br>
Abhishek Lalwani (alalwani@umass.edu) <br>
Himanshu Gupta (hgupta@umass.edu) <br>
Rushikesh Dudhat (rdudhat@umass.edu) <br>

# System requirements

GPU </br>
Cuda 11/11.2 (should work on 12 as well but we did not get a chance to test on it) </br>
MSVC 2019 Build Tools </br>
Windows (The code will mostly work on linux/Mac systems as well but we have done extensive testing in Windows and Google Collab Notebooks). </br>

# Setting up the environment
1. Activate the environment in which you want to test our code.
2. Make sure your CUDA_PATH variable is set up.
3. `pip install -r requirements.txt`
4. `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
5. `cd neural-renderer-master`
6. `python setup.py install --user`
7. `pip freeze` (neural-renderer-pytorch==1.1.3 should be installed in your environment)
8. `cd ../style_transfer_3d-master`
9. `python setup.py install --user`
10. `pip freeze` (style-transfer-3d==0.0.1 should be installed in your environment)
11. `python ./examples/run.py -im examples/data/meshes/bunny.obj -is examples/data/styles/gogh2.jpg -o examples/data/results/bunny_gogh2.gif -rd examples/data/results (Windows specific command)` or `bash ./examples/run.sh` (Linux Specific command)
12. You can check the results in the examples/data/results folder. Output gif name will be `bunny_gogh2.gif` (for windows specific command). Please note that we also apply texture mapping to our inputs for further results and the output for that will be `stylized_and_textured_bunny_gogh2.gif`. Similar naming strucutre can also be used to check results for Linux systems.





