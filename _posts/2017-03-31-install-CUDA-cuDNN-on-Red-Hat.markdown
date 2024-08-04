---
layout: post
title: "Install CUDA and cuDNN on Red Hat"
date: 2017-03-30 22:39:51 -0500
categories: tools
---

I've been building neural networks for my chemical space deep learning research since last year. Speedup of training is always one of the central topics. Recently my research group purchased a `Quadro K2200` for our Red Hat workstation. I thought it's a good opportunity of accelerating the computation by switching to GPU. The benefit detail for my research projects will probably be covered in later posts.

Today I'm going to focus on how to smoothly install the **two important tools** for any neural network applications to run on GPUs: **CUDA** and **cuDNN**. I noticed there's serveral installation guides online, e.g., [NVIDIA official guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4cwRN8XWN), [AWS EC2 guide 1](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/) and [AWS EC2 guide 2](http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/). They either are outdated, missing some key steps or contain unneccessary settings. More importantly for beginners, we want to have some tests to see each major step is gone though correctly and neccessarily.

For this purpose I decided to create this post, whose goal is to install CUDA and cuDNN on Red Hat Enterprise Linux 7 in a more **transparent** and **reasonable** way.

Just to emphasize, my situation was:

- I could easily install `theano`/`tensorflow`/`keras` through `anaconda` binary platform,

- my application can already successfully run on CPUs,

- I only need to make `theano`/`tensorflow`/`keras` detect there's GPU available

## Test Examples Prep

I prepared two python test scripts: `example_1` is from [theano official documentation](http://deeplearning.net/software/theano/tutorial/using_gpu.html), which is easy and fast to test whether we have connected to GPU. `example_2` is from [keras example cifar10_cnn](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py), which will be used to final check the speedup brought by GPU. Below is the detail of `example_1` script.

{% highlight python %}
from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 _ 30 _ 768 # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
('Gpu' not in type(x.op).__name__)
for x in f.maker.fgraph.toposort()]):
print('Used the cpu')
else:
print('Used the gpu')
{% endhighlight %}

## Python Anaconda Environment Setup

One command can create a conda environment with: `theano`. Just run on your terminal:
{% highlight bash %}
$ conda create -n theano_test -c conda-forge theano
{% endhighlight %}

# Status of test examples: able to run on CPUs but not GPUs

At this point, we'll run `example_1` script (no need to run `example_2`) to make sure `example_1` can run on CPUs but not on GPUs.

Run `example_1` on CPU mode:
{% highlight bash %}
(theano_test) [kehang]$ python example_1.py
{% endhighlight %}

Result of `example_1` for CPU mode:
{% highlight bash %}
[Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
Looping 1000 times took 4.095498 seconds
Result is [ 1.23178032 1.61879341 1.52278065 ..., 2.20771815 2.29967753
1.62323285]
Used the cpu
{% endhighlight %}

Run `example_1` on GPU mode:
{% highlight bash %}
(theano_test) [kehang]$ THEANO_FLAGS=device=cuda python example_1.py
{% endhighlight %}

Result of `example_1` for GPU mode:
{% highlight bash %}
ERROR (theano.gpuarray): Could not initialize pygpu, support disabled
Traceback (most recent call last):
File "/home/kehang/miniconda/envs/keras_test/lib/python2.7/site-packages/theano/gpuarray/**init**.py", line 164, in <module>
use(config.device)
File "/home/kehang/miniconda/envs/keras_test/lib/python2.7/site-packages/theano/gpuarray/**init**.py", line 151, in use
init_dev(device)
File "/home/kehang/miniconda/envs/keras_test/lib/python2.7/site-packages/theano/gpuarray/**init**.py", line 60, in init_dev
sched=config.gpuarray.sched)
File "pygpu/gpuarray.pyx", line 614, in pygpu.gpuarray.init (pygpu/gpuarray.c:9415)
File "pygpu/gpuarray.pyx", line 566, in pygpu.gpuarray.pygpu_init (pygpu/gpuarray.c:9106)
File "pygpu/gpuarray.pyx", line 1021, in pygpu.gpuarray.GpuContext.**cinit** (pygpu/gpuarray.c:13468)
GpuArrayException: Error loading library: -1
[Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
Looping 1000 times took 4.158732 seconds
Result is [ 1.23178032 1.61879341 1.52278065 ..., 2.20771815 2.29967753
1.62323285]
Used the cpu
{% endhighlight %}

## Installation of CUDA Toolkit and Driver

This part is well documented by [NVIDIA official guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4cwRN8XWN) except that some small steps are either too brief (can be not so actionable) or outdated. This post will cover all the commands step by step in an actionable way, for detailed explanations one can always refer to the official guide.

# Pre-installation Actions

This step is 100% following [NVIDIA official guide: Pre-installation Actions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions). First run the following commands to verify system requirements are met:

{% highlight bash %}

# Verify You Have a CUDA-Capable GPU

$ lspci | grep -i nvidia

# Verify You Have a Supported Version of Linux

$ uname -m && cat /etc/\*release

# Verify the System Has gcc Installed

$ gcc --version

# Install Correct Kernel Headers and Development Packages for Red Hat

$ sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)

{% endhighlight %}

Now you go to [NVIDIA CUDA Toolkit Downloads](http://developer.nvidia.com/cuda-downloads). Below is what I chose for my Red Hat EL7 machine. It's a fresh installation so I didn't have to deal with conflicting previous installations. But for people having previous CUDA installation, please refer to [Handle Conflicting Installation Methods](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#handle-uninstallation).

{:.cuda_cudnn_img}
![Alt text]({{ site.github.url }}/assets/cuda_cudnn_post/img/cuda_download.png){:width="80%"}

# Package Manager Installation

As I mentioned early, I chose to download the `rpm(local)` installer option, so I need to use package manager installation. This step we'll following 80% of the [NVIDIA official guide: Package Manager Installation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) and add/modify some steps I regard neccessary but not clear in the official guide.

For users choosing `runfile`, please refer to [Runfile Installation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile).

{% highlight bash %}

# Add third-party repository EPEL to yum repolist

$ wget http://dl.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-9.noarch.rpm
$ rpm -ivh epel-release-7-9.noarch.rpm

# Verify you have EPEL now

$ yum repolist

# Search in EPEL for dkms and libvdpau

# which are dependencies of CUDA

$ yum --enablerepo=epel info dkms
$ yum --enablerepo=epel info libvdpau

# Address custom xorg.conf, if applicable

# I don't have xorg.conf before so it's fine

# if you have, please follow official guide

# Install cuda finally

# takes 10 mins or so

$ sudo rpm -i cuda-repo-rhel7-8-0-local-ga2-8.0.61-1.x86_64.rpm
$ sudo yum clean all
$ sudo yum install cuda

{% endhighlight %}

# Post-installation Actions

You only need to add `/usr/local/cuda-8.0/bin` (version number can vary, please check) to your `PATH` environment variable (either by doing as in below or put into your `.bashrc` file).

{% highlight bash %}
$ export PATH=/usr/local/cuda-8.0/bin:$PATH
{% endhighlight %}

To verify GPUs can be accessed, three small steps are needed

- restart the machine

- verify the driver version by running `cat /proc/driver/nvidia/version`

- run `deviceQuery` cuda sample binary, in steps as follow
  {% highlight bash %}

# Copy CUDA samples to your personal directory

# so that you have write permission

$ cuda-install-samples-8.0.sh <your-target-directory>
$ cd <your-target-directory>/NVIDIA_CUDA-8.0_Samples

# Compile samples

$ make

# Running deviceQuery

$ cd bin/x86_64/linux/release
$ ./deviceQuery

{% endhighlight %}

Here's what you'll get after running `deviceQuery`:

{:.cuda_cudnn_img}
![Alt text]({{ site.github.url }}/assets/cuda_cudnn_post/img/deviceQuery.png){:width="80%"}

# Status of test examples: able to run on GPUs but cuDNN complaints

Run `example_1` on GPU mode:
{% highlight bash %}
(theano_test) [kehang]$ THEANO_FLAGS=device=cuda python example_1.py
{% endhighlight %}

Result of `example_1` for GPU mode:

{% highlight bash %}
Can not use cuDNN on context None: cannot compile with cuDNN. We got this error:
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status

Mapped name None to device cuda: Quadro K2200 (0000:03:00.0)
[GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
Looping 1000 times took 0.535818 seconds
Result is [ 1.23178032 1.61879341 1.52278065 ..., 2.20771815 2.29967753
1.62323285]
Used the gpu
{% endhighlight %}

Hooray! `example_1` is able to access to GPU `Quadro K2200` and the wallclock has been reduced by a factor of 8 (from 4.15 sec to 0.53 sec.)

But also it shows that `cannot find -lcudnn`, which will be our next installation part: **cuDNN installation**.

## Installation of cuDNN

This part of installation is relatively easy, and we'll mainly follow [AWS EC2 guide 2](http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/). But still some steps of it need modified or better explained.

# cuDNN Download

To obtain the cuDNN library, one needs

- create an account to join [NVIDIA developer program](https://developer.nvidia.com/accelerated-computing-developer).
- download [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)

{:.cuda_cudnn_img}
![Alt text]({{ site.github.url }}/assets/cuda_cudnn_post/img/cudnn_download.png){:width="80%"}

I chose **`cuDNN Library v5.1 for Linux`** not `v6.0` is because latest `theano` can only utilize up to `v5.1` (`v6.0` has a conflict with `theano`, but maybe future `theano` can cooperate)

# cuDNN Unpack and Install

Copy the download onto your machine, and unpack it
{% highlight bash %}
$ tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
{% endhighlight %}

Copy cuDNN header file and library files to appropriate `include` and `lib64` directories.
{% highlight bash %}

# change directory to the unpacked folder of cudnn

$ cd cuda

# copy related files to /usr/local/cuda/lib64 or /usr/local/cuda/include

# -av will keep the symbolic links as is during copying

sudo cp -av lib64/_ /usr/local/cuda/lib64/
sudo cp -av include/_ /usr/local/cuda/include/

# Update your environment variables in bash session

# or put them in your .bashrc file

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:$LIBRARY_PATH

{% endhighlight %}

# Status of test examples: run on GPUs without any complaints

Run `example_1` on GPU mode:
{% highlight bash %}
(theano_test) [kehang]$ THEANO_FLAGS=device=cuda python example_1.py
{% endhighlight %}

Result of `example_1` for GPU mode:

{% highlight bash %}
Using cuDNN version 5110 on context None
Mapped name None to device cuda: Quadro K2200 (0000:03:00.0)
[GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
Looping 1000 times took 0.536293 seconds
Result is [ 1.23178032 1.61879341 1.52278065 ..., 2.20771815 2.29967753
1.62323285]
Used the gpu
{% endhighlight %}

Although `example_1` doesn't show time advantage of using `cuDNN` but it clearly shows it's using cuDNN smoothly.

For `example_2`, which is a convolutional neural network application, it can run on GPUs as well (below right-corner video). After switching to GPUs, the training process is sppeded up by at least 6 times (from 291 sec/epoch to 45 sec/epoch, I guess you can immediately tell from the progress bar)

<iframe width="360" height="215" src="https://www.youtube.com/embed/14sQNwBFv9s" frameborder="0" allowfullscreen></iframe>

<iframe width="360" height="215" src="https://www.youtube.com/embed/rO1qwGVB47w" frameborder="0" allowfullscreen></iframe>

But anyways, hope you can enjoy the installation guide of CUDA and cuDNN on Red Hat. You can leave your comments on Youtube if you have any questions or suggestions.
