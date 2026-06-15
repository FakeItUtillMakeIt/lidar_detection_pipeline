lidar detection pipeline

build:
    export CUDA_Inc=/usr/local/cuda-13.1/include/ && export CUDA_Lib=/usr/local/cuda-13.1/lib64/ && export TensorRT_Inc=/usr/include/x86_64-linux-gnu/ && export TensorRT_Lib=/usr/lib/x86_64-linux-gnu/ && cd /home/sevnce/lj/project/lidar_detection_pipeline/build && rm -rf * && cmake .. && make -j$(nproc) 2>&1

run:
    ./lidar_app --input ../data --model ../model/pointpillar.plan --output-types file --output ../out --timer 2>&1 