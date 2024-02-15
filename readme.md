# Summary

This is a simple project made with the sole purpose of studying development process in [CUDA](https://developer.nvidia.com/cuda-toolkit). This project offers a console application that takes two matrices of floating-point numbers and calculates their multiplication.

## Requirements

[Cuda Toolkit 12.3](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)


## Running:

```shell
nvcc matrix_mult.cu
./a.out
```

Similarly, to execute certain test cases, this can be accomplished using the following command.

```shell
./a.out < test/inputs/input_4.txt
```