docker pull rocm/pytorch:latest

docker run -it --network=host -v /home/yigex/huggingface/heyi:/workspace  --device=/dev/kfd --device=/dev/dri --group-add video  --security-opt seccomp=unconfined --ipc=host --name heyi_apex_wx  rocm/pytorch:latest

mkdir heyi && cd heyi

git clone https://github.com/eliotwang/apex

cd apex

python setup.py install --cpp_ext --cuda_ext

cd tests

bash run.sh