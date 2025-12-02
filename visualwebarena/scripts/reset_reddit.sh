#!/bin/bash

sudo docker stop forum
sudo docker rm $(sudo docker ps -a | grep forum | awk '{print $1}')
sudo docker run --name forum -p 9999:80 -p 9998:5432 -d postmill-populated-exposed-withimg
# wait ~15 secs for all services to start
sleep 15
