install:
	wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
	./cuda_12.3.2_545.23.08_linux.run
	apt update && apt upgrade -y
	apt install -y nvidia-cuda-toolkit

run:
	nvcc -o exe main.cu
	./exe
	rm -rf exe