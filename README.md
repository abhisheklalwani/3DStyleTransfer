# 674-Final-Project

This repository contains implementation of the Final Project of COMPSCI 674 course at UMass Amherst. <br>
Contributors - <br>
Abhishek Lalwani (alalwani@umass.edu) <br>
Himanshu Gupta (hgupta@umass.edu) <br>
Rushikesh Dudhat (rdudhat@umass.edu) <br>

# System requirement

GPU </br>
Cuda 11.2 </br>
torch </br>
numpy </br>
skimage </br>
tqdm </br>
imageio </br>
glob </br>
matplotlib </br>
cv2 </br>
torchvision </br>

# Installing Required Packages  


Installing Docker
1. Follow the steps specified in the link: https://docs.docker.com/engine/installation/

# Important Source File Descriptions
1. `catalog/catalog.py` implements the catalog server with the relevant GET and PUT methods.
2. `frontend/frontend.py` implements the front end server with the buy, search and lookup methods.
3. `order/order.py` implements the order server with the buy method. 
4. `Docs` folder contains the design documentation and API documention.
5. `logs` folder contains all the logs of the servers after the execution of the script `runme.sh`. For eg. orderA.log, frontend.log etc. It will also contain the log file `heartbeat.log` which contains the heartbeats of different servers captured by frontend.
6. `requirements.txt` contains the python libraries required.
7. `env.cfg` contains the `PUBLIC_IP` and `PORT` of the machines where the catalog, order and frontend server has to be run. It also contains the reference to the `pem` file that is required to ssh, and scp to the remote machines. Modify the file according to your requirements.
8. `runme.sh` is a single script to automatically deploy catalog, order and frontend docker servers on specified machines in `env.cfg`, run the client.py and get the logs from all the servers.
9. `client.py` starts the traffic by sending multiple requests to frontend parrallely and sequentially. This script is called by `runme.sh` internally.
10. `simulate_fault_tolerance.sh` is a single script to check falut tolerance of the system. The script specifically brings down the catalogA and invokes the client.py to check if the system is working correctly even if one of the server is down. Then it brings the server up and calls the python file `test_server_recovery.py` to check if the system was able to recover from the crash. 
11. `const.py` contains information about the books in the catalog server.
12. `Docs` contains the design documentation 

Please find the instructions below for testing the implementation.

# Instructions 

### To run the server locally

1. Define the ip and ports of order, catalog and front end server by editing the `env.cfg` file. For running locally make the IPs for the server as `http://<public_ip_of_local_vm>`.
2. Now, on your local machine, run the `runme.sh` file which will deploy all the dockers and then trigger the client.py for starting the traffic. USAGE: `. ./runme.sh`. If you face any permission issue, try to run the script with root user.
3. You can observe the results of this run in different log files that should be accumulated under the folder `logs`. `client.log` in the main folder will contain the logs of `client.py` script.

### To run servers remotely 

1. Deploy an Ubuntu VM. We have used the AMI: `ami-013f17f36f8b1fefb` to deploy instances and test our code. You can use any other Ubuntu image as per your convenience. Make sure to install docker engine on the instances. You can follow the link: https://docs.docker.com/engine/installation/ for the same.
2. Get the private .pem file which will be used to coomunicate to the remote servers by the local machine.
3. Edit the security group to ensure that the ports required by the peers to communicate are open.
3. Set up password-less ssh from the local machine to the ec2 servers by running the following command from the local terminal:
    `ssh -i <pem_file_path> ubuntu@<ec2_public_ip>` with the private pem file and the public IP address of the EC2 instance that has been set up. (This will add the ec2 server to the known_host file so that you can ssh from the script without the need of a password). Please note that ubuntu is default username of an AWS Ubuntu VM. You can alter it according to your need.
4. Define the ip and ports of order, catalog and front end server by editing the `env.cfg` file. Change the port IDs of the servers as pleased. Also provide the path of the pem file. Instructions to change the file is specified in the same. Make sure all the remote server can be accessed with the same file.
5. Now, on your local machine, run the `runme.sh` file which deploy all the dockers and then trigger the client.py for starting the traffic. USAGE: `. ./runme.sh`.
6. You can observe the results of this run in different log files that should be accumulated under the folder `logs`. `client.log` in the main folder will contain the logs of `client.py` script.

### To simute and check fault tolerance of the system

1. Execute the shell script `simulate_fault_tolerance.sh` using the command `. ./simulate_fault_tolerance.sh`.
2. Check the the log file `test_server_recovery.log` to check if the system was able to recover from the crash of catalogA server.
