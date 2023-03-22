rm -rf ../tmp/*.npy

rm -rf ./results/*

python hat/test.py -opt HAT_GAN_Real_SRx4.yml

python rename.py
