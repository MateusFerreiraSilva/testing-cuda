# Summary

This is a simple project made with the sole purpose of studying development process in [CUDA](https://developer.nvidia.com/cuda-toolkit).

## Requirements

[Cuda Toolkit 12.3](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)


## Running:

```shell
nvcc -g -G hello-world.cu -o hello-world # compilation, -g -G are debug tags
./hello-world
```