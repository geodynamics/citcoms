#!/bin/sh

# Credits to deadsnakes and https://github.com/likueimo/ubuntu_gmt4

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.3
sudo apt-get install -y python3-pip
python3.3 -m pip install numpy==1.7.1
python3.3 -m pip install scipy==0.12.0
git clone https://github.com/likueimo/ubuntu_gmt.git
cd ubuntu_gmt
sudo bash install_bash_script
echo 'export GMT4HOME=/opt/GMT-4.5.14' >> ~/.bashrc
echo 'export PATH=${GMT4HOME}/bin:$PATH' >> ~/.bashrc
