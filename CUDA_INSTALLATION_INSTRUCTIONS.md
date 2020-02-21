# Setting Up GPU Support on Azure (Or any cloud)

[![ubuntu-16.04](https://img.shields.io/static/v1?label=Ubuntu&message=16.04&color=success)](http://releases.ubuntu.com/16.04/)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![cuda-9.0](https://img.shields.io/static/v1?label=CUDA&message=9.0&color=blue)](https://developer.nvidia.com/cuda-90-download-archive)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![cudnn-7.x](https://img.shields.io/static/v1?label=cuDNN&message=7.x&color=blueviolet)](https://developer.nvidia.com/rdp/cudnn-archive)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![nvidia-drivers](https://img.shields.io/static/v1?label=NVIDIA-Drivers&message=384&color=important)](https://www.nvidia.com/download/driverResults.aspx/123335/en-us)

First, create cloud account for starting development of project. Create Virtual Machine Instance with following specifications.
  - Ubuntu 16.04
  - GPU must be there. (Don't use General Purpose instance)
  - Atleast 2vCPUs and 7.5 GB RAM.

Connect the VM instance through terminal. (If the cloud you are using don't provide terminal then use [puTTY](https://www.ssh.com/ssh/putty)).

## Installing NVIDIA Graphics Drivers
Note here that CUDA-9.0 supports NVIDIA Drivers with version 384 and above.
```sh
$ sudo apt-get update
$ sudo apt-get install nvidia-384 nvidia-modprobe
$ sudo apt-get update
```
After executing above commands, you can check whether the installation is complete or not with the command `nvidia-smi`. If the installation is complete then it will show the table corresponding to particular NVIDIA Driver.

## Installing CUDA-9.0
Installing CUDA from runfile is actually a breeze, compared to apt-get which involves adding NVIDIA repos and messing up your configuration.
The CUDA runfile installer can be downloaded from [NVIDIA’s website](https://developer.nvidia.com/cuda-downloads), or using wget in case you can’t find it easily on NVIDIA:
(Note: the version I downloaded at first was 384.81 but NVidia constantly releases new minor versions. I recommend the latest which is 384.130 as of 8/29/2018)
```sh
$ cd
$ wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
```
What you download is a package the following three components:
  - An NVIDIA driver installer, but usually of stale version
  - The actual CUDA installer
  - The CUDA samples installer

I suggest extracting the above three components and executing 2 and 3 separately (remember we installed the driver ourselves already). To extract them, execute the runfile installer with --extract option:
```sh
$ chmod +x cuda_9.0.176_384.81_linux-run
$ ./cuda_9.0.176_384.81_linux-run --extract=$HOME
```
You should have unpacked three components: 
  - NVIDIA-Linux-x86_64-384.81.run (1. NVIDIA driver that we ignore)
  - cuda-linux.9.0.176-22781540.run (2. CUDA 9.0 installer)
  - cuda-samples.9.0.176-22781540-linux.run (3. CUDA 9.0 Samples).

Execute the second one to install the CUDA Toolkit 9.0:
```sh
$ sudo ./cuda-linux.9.0.176-22781540.run
```
You now have to accept the license by scrolling down to the bottom (hit the **d** key on your keyboard) and enter **accept**. Next accept the defaults.
To verify our CUDA installation, install the sample tests by:
```sh
$ sudo ./cuda-samples.9.0.176-22781540-linux.run
```
After the installation finishes, configure the runtime library.
```sh
$ sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
$ sudo ldconfig
```
t is also recommended for Ubuntu users to append string `/usr/local/cuda/bin` to system `file /etc/environment` so that nvcc will be included in `$PATH`. This will take effect after reboot. To do that, you just have to
```sh
$ sudo vim /etc/environment
```
and then add `:/usr/local/cuda/bin` (including the ":") at the end of the `PATH="/blah:/blah/blah"` string (inside the quotes).
After a reboot, let's test our installation by making and invoking our tests:
```sh
$ cd /usr/local/cuda-9.0/samples
$ sudo make
```
It’s a long process with many irrelevant warnings about deprecated architectures (`sm_20` and such ancient GPUs).
**Note** : If above command gives the error about the version conflict of `GNU` or `gcc` or `g++` then install `gcc-6` and `g++-6` first and then try again. **CUDA-9.0 requires gcc version 6 or below**. 
After it completes, run `deviceQuery` and `p2pBandwidthLatencyTest`:
```sh
$ cd /usr/local/cuda/samples/bin/x86_64/linux/release
$ ./deviceQuery
```
After running `deviceQuery`, the system will check the installation and `RESULT=PASS` will be shown on screen.
**Cleanup** : if `./deviceQuery` works, remember to rm the 4 files (1 downloaded and 3 extracted).

## Install cuDNN-7.x
The recommended way to install cuDNN 7.x is to download all 3 `.deb` files. I had previously recommended using the `.tgz` installation approach, but found out that it didn’t allow verification by running code samples (no way to install the code samples `.deb` after `.tgz` installation).
Downloading the cuDNN file from [NVIDIA cuDNN website](https://developer.nvidia.com/rdp/cudnn-download) can be very frustrating if you are using cloud VM through terminal because you can only download the file to your computer. You can't download it in VM by `wget` because it changes the configuration of `.deb` or `.tgz` file. 
But the solution here is we can download raw cuDNN `.deb` files from [NVIDIA Repo](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/) like we download raw `.whl` files for pip installing package.
You can choose the link corresponding to the version of your preference.
  - Installing runtime library first
    ```sh
    $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
    $ chmod +x libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
    $ sudo -dpkg -i libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
    ```
  - Installing developer library
    ```sh
    $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.1.3.16-1+cuda9.0_amd64.deb
    $ chmod +x libcudnn7-dev_7.1.3.16-1+cuda9.0_amd64.deb
    $ sudo -dpkg -i libcudnn7-dev_7.1.3.16-1+cuda9.0_amd64.deb
    ```
  - Installing code samples. Note here that this file is of just 4 MB. So, if this doesn't work then you can download this file in your computer, upload it to github and clone it in your cloud VM.
    ```sh
    $ wget http://apt.cs.uchicago.edu/pool/main/ c/cudnn/ libcudnn7-doc_7.1.3.16-1+cuda9.0_amd64.deb
    $ chmod +x libcudnn7-doc_7.1.3.16-1+cuda9.0_amd64.deb
    $ sudo -dpkg -i libcudnn7-doc_7.1.3.16-1+cuda9.0_amd64.deb
    ```

Now we can verify the cuDNN installation (below is just the official guide, which surprisingly works out of the box):
  - Copy the code samples somewhere you have write access: `cp -r /usr/src/cudnn_samples_v7/ ~`
  - Go to the MNIST example code: `cd ~/mnistCUDNN`. Path may change w.r.t. version of cuDNN you have installed.
  - Compile the MNIST example: `make clean && make`.
  - Run the MNIST example: `./mnistCUDNN`. If your installation is successful, you should see `Test passed!` at the end of the output.

## Do NOT Install cuda-command-line-tools
Contrary to the official TensorFlow installation docs, you don’t need to install `cuda-command-line-tools` because it’s already installed in this version of CUDA and cuDNN. If you **apt-get** it, you won’t find it.

## Configure the CUDA and cuDNN library paths
What you do need to do, however, is exporting environment variables `LD_LIBRARY_PATH` in your `.bashrc` file:
```sh
# put the following line in the end or your .bashrc file
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"
```
And source it by `source ~/.bashrc`.

## References
  - [Zhanwen Chen's Medium blog on CUDA Installation](https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)
  - [Harvey's answer on stackExchange](https://askubuntu.com/questions/1031197/cudnn-installation-on-ubuntu-server-16-04)
  - [NVIDIA compute repo for downloading raw files](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/)
  - [Azure Documentation](https://docs.microsoft.com/en-us/azure/)
