This document shows how to first use gCloud to make an instance, and then to run
large-scale experiments.

- [gCloud Setup](#gcloud)
- [Commands](#commands)
    - [Reproducing Transporter Nets Results](#reproducing-transporter-results)
    - [New Results and Commands](#new-results)
- [Data Management](#data-management)
    - [Test Rewards](#test-rewards)
    - [Neural Networks](#neural-networksmanagement)
    - [Tensorboard](#tensorboard)


# gCloud

Here are instructions on how to reproduce results from Transporters.

(1) Make a new VM instance using the GUI on GCP. Unfortunately it takes a while
before machines with GPUs are available.

(2) ssh into the machine via the click interface.

(3) Git clone this repository.

(4) Run: `bash install_dependencies.sh` then `. ~/.bashrc`.

(5) Download any data as needed from our storage bucket, such as:

```
gsutil -m cp -r gs://research-brain-deformable-manipulation-xgcp/data/cloth-flat-heuristic-01.tar.gz .
```

(6) **If necessary, set up an X server on the virtual machine**. This is
necessary to get cloth to render correctly on Ubuntu 18.04 machines from gCloud.
For this, [follow the instructions that I described here for Blender][3]. Enter
GNU screen to do:

```
sudo Xorg :1
```

and leave it running forever (check with `nvidia-smi` that it exists). Then in
the same screen, do:

```
export DISPLAY=:1
```

and we *should* be able to run PyBullet with `--disp` on. Note that my VMs are
created with the option for a display, perhaps that is why the above worked?

(7) Run any of the commands I want to test results. I recommend using the
"insertion" environment as it proceeds quickly. Use the Transporter agent to
check that TensorFlow is working.

Note: it will print results like this:

```
Train Iter: XX, Loss: YY ZZ
```

The first number is the iteration, the second is the loss, and sometimes there's
a third (e.g., for transporters) because we have the attention and then the
transport loss.

If there's a problem with the machine just follow these steps:

```
sudo apt-get --purge -y remove 'cuda*'
sudo apt-get --purge -y remove 'nvidia*'
```

and try again. :)

You shouldn't need to do this, but [here is a nice guide on symlinking][2] with
multiple CUDA versions.

**Example run with a Tesla P4 GPU**: this is the message TensorFlow gives me
when I initially run a command on gCloud machines:

```
seita@danielseita-experimental-03:~/pybullet-def-envs$ python main.py --gpu=0 --agent=transporter --task=insertion --num_demos=1
2020-07-17 20:55:53.136732: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/lib64
2020-07-17 20:55:53.136863: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/lib64
2020-07-17 20:55:53.136880: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
pybullet build time: Jul  8 2020 18:24:12
Detected TensorFlow version:  2.1.0
2020-07-17 20:55:54.222023: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-07-17 20:55:54.241648: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:55:54.242293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:00:04.0 name: Tesla P4 computeCapability: 6.1
coreClock: 1.1135GHz coreCount: 20 deviceMemorySize: 7.43GiB deviceMemoryBandwidth: 178.99GiB/s
2020-07-17 20:55:54.242716: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-07-17 20:55:54.245729: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-17 20:55:54.248340: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-07-17 20:55:54.248982: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-07-17 20:55:54.251172: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-07-17 20:55:54.252747: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-07-17 20:55:54.261661: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-17 20:55:54.261784: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:55:54.262470: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:55:54.263159: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
```

I don't think any of the warnings matter to me. I don't think we use TensorRT.
Once we start *training*:

```
2020-07-17 20:57:22.713063: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-17 20:57:22.719288: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2000175000 Hz
2020-07-17 20:57:22.719672: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5649039740e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-17 20:57:22.719702: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-07-17 20:57:22.821342: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.821969: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56490394c460 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-07-17 20:57:22.822007: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P4, Compute Capability 6.1
2020-07-17 20:57:22.822231: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.822733: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
pciBusID: 0000:00:04.0 name: Tesla P4 computeCapability: 6.1
coreClock: 1.1135GHz coreCount: 20 deviceMemorySize: 7.43GiB deviceMemoryBandwidth: 178.99GiB/s
2020-07-17 20:57:22.822836: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-07-17 20:57:22.822857: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-17 20:57:22.822881: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-07-17 20:57:22.822912: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-07-17 20:57:22.822947: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-07-17 20:57:22.822972: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-07-17 20:57:22.823015: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-17 20:57:22.823108: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.823638: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.824039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2020-07-17 20:57:22.824101: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-07-17 20:57:22.825193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-17 20:57:22.825220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0
2020-07-17 20:57:22.825228: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N
2020-07-17 20:57:22.825374: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.825887: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-17 20:57:22.826375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7131 MB memory) -> physical GPU (device: 0, name: Tesla P4, pci bus id: 0000:00:04.0, compute capability: 6.1)
Exception in thread Thread-2:
Traceback (most recent call last):
  File "/home/seita/miniconda3/lib/python3.7/threading.py", line 926, in _bootstrap_inner
    self.run()
  File "/home/seita/miniconda3/lib/python3.7/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/home/seita/pybullet-def-envs/ravens/environment.py", line 62, in step_simulation
    p.stepSimulation()
pybullet.error: Not connected to physics server.

2020-07-17 20:57:25.131787: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-17 20:57:25.912984: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-17 20:57:26.426684: I tensorflow/core/kernels/cuda_solvers.cc:159] Creating CudaSolver handles for stream 0x5649065ce150
2020-07-17 20:57:26.426800: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-07-17 20:57:26.695511: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
Train Iter: 0 Loss: 10.3523 14.0204
Train Iter: 1 Loss: 10.3906 14.0236
Train Iter: 2 Loss: 10.3621 14.0185
Train Iter: 3 Loss: 10.2591 14.0190
```

I am not sure if the PyBullet error should matter here, since we are training
and not testing. I don't get any notable errors. For GPUs with Transporters:

- Tesla P4s should be seeing very high GPU utilization (at least 50%, closer to
  100%).
- Tesla P100s should be seeing around 50% GPU utilization on average.


# Commands

## Reproducing Transporter Results

For each of the environments, run with `--num_demos` as 1, 10, 100, and 1000.
[See this document][1] for the reference. *Always check the `data/` directory
because that has the demonstration data*.

```
python main.py --gpu=0 --agent=transporter --task=hanoi --num_demos=1
python main.py --gpu=0 --agent=transporter --task=hanoi --num_demos=10
python main.py --gpu=0 --agent=transporter --task=hanoi --num_demos=100
python main.py --gpu=0 --agent=transporter --task=hanoi --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=sorting --num_demos=1
python main.py --gpu=0 --agent=transporter --task=sorting --num_demos=10
python main.py --gpu=0 --agent=transporter --task=sorting --num_demos=100
python main.py --gpu=0 --agent=transporter --task=sorting --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=insertion --num_demos=1
python main.py --gpu=0 --agent=transporter --task=insertion --num_demos=10
python main.py --gpu=0 --agent=transporter --task=insertion --num_demos=100
python main.py --gpu=0 --agent=transporter --task=insertion --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=aligning --num_demos=1
python main.py --gpu=0 --agent=transporter --task=aligning --num_demos=10
python main.py --gpu=0 --agent=transporter --task=aligning --num_demos=100
python main.py --gpu=0 --agent=transporter --task=aligning --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=stacking --num_demos=1
python main.py --gpu=0 --agent=transporter --task=stacking --num_demos=10
python main.py --gpu=0 --agent=transporter --task=stacking --num_demos=100
python main.py --gpu=0 --agent=transporter --task=stacking --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=sweeping --num_demos=1
python main.py --gpu=0 --agent=transporter --task=sweeping --num_demos=10
python main.py --gpu=0 --agent=transporter --task=sweeping --num_demos=100
python main.py --gpu=0 --agent=transporter --task=sweeping --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=cable --num_demos=1
python main.py --gpu=0 --agent=transporter --task=cable --num_demos=10
python main.py --gpu=0 --agent=transporter --task=cable --num_demos=100
python main.py --gpu=0 --agent=transporter --task=cable --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=palletizing --num_demos=1
python main.py --gpu=0 --agent=transporter --task=palletizing --num_demos=10
python main.py --gpu=0 --agent=transporter --task=palletizing --num_demos=100
python main.py --gpu=0 --agent=transporter --task=palletizing --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=kitting --num_demos=1
python main.py --gpu=0 --agent=transporter --task=kitting --num_demos=10
python main.py --gpu=0 --agent=transporter --task=kitting --num_demos=100
python main.py --gpu=0 --agent=transporter --task=kitting --num_demos=1000

python main.py --gpu=0 --agent=transporter --task=packing --num_demos=1
python main.py --gpu=0 --agent=transporter --task=packing --num_demos=10
python main.py --gpu=0 --agent=transporter --task=packing --num_demos=100
python main.py --gpu=0 --agent=transporter --task=packing --num_demos=1000
```

## New Results

For cloth, **with 480 Hz time step**, AND the **`--disp` option** because we
need that for cloth to be visible on headless VMs (for now):

```
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat-easy --num_demos=1
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat-easy --num_demos=10
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat-easy --num_demos=100
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat-easy --num_demos=1000

python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat --num_demos=1
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat --num_demos=10
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat --num_demos=100
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-flat --num_demos=1000

python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-cover --num_demos=1
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-cover --num_demos=10
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-cover --num_demos=100
python main.py --gpu=0 --agent=transporter --hz=480 --disp --task=cloth-cover --num_demos=1000
```

# Data Management

I save demonstrator data files as
`pybullet-def-envs/data/[env]-1000-demos.tar.gz`. To *upload* them from a
virtual machine to my bucket, use commands like this:

```
gsutil cp data/*-1000-demos.tar.gz gs://research-brain-deformable-manipulation-xgcp/data/
```

To *download* them from my bucket to a virtual machine, use:

```
gsutil cp gs://research-brain-deformable-manipulation-xgcp/data/*-1000-demos.tar.gz data/
```

If you get this error when uploading files:

```
ResumableUploadAbortException: 403 Insufficient Permission
```

then run `gcloud auth login` on the *remote / virtual machine*. It should say that I am logged
in as "seita@google.com" with project "deformable-manipulation-xgcp".

Results from `main.py` are saved in three primary places.

## Test Rewards

Suppose we've run "sorting" with the "transporter" agent for all of the four
demos. Then we get:

```
sorting-transporter-1-0.pkl
sorting-transporter-10-0.pkl
sorting-transporter-100-0.pkl
sorting-transporter-1000-0.pkl
```

(The last number is the training run, because the formal paper should have 3
runs per setting for more statistical significance.) These pickle files are for
lightweight statistics; they are simply a list with tuple items, where the items
are `(train_iters, total_reward)`, which is collected *once* per test episode.
By default we do 20 test episodes, so there are 20 items with the same
"train_iter" in them. Just take the average of those 20 for performance.


## Neural Networks

For the *neural networks*, we have the checkpoints folder:

```
seita@danielseita-experimental-03:~/pybullet-def-envs$ ls -lh checkpoints/
drwxrwxr-x 2 seita seita 4.0K Jul 19 20:43 sorting-transporter-1-0
drwxrwxr-x 2 seita seita 4.0K Jul 19 06:40 sorting-transporter-10-0
drwxrwxr-x 2 seita seita 4.0K Jul 19 00:47 sorting-transporter-100-0
drwxrwxr-x 2 seita seita 4.0K Jul 18 14:30 sorting-transporter-1000-0
```

Each of the directories above will contain checkpoints of the neural networks.
With transporters, we have an *attention* and a *transport* net, and testing was
done every 1000 training iterations. Therefore, the files look like this for one
directory:

```
seita@danielseita-experimental-03:~/pybullet-def-envs$ ls -lh checkpoints/sorting-transporter-1-0/
total 100M
-rw-rw-r-- 1 seita seita 1.7M Jul 19 13:54 attention-ckpt-1000.h5
-rw-rw-r-- 1 seita seita 1.7M Jul 19 17:09 attention-ckpt-10000.h5
-rw-rw-r-- 1 seita seita 1.7M Jul 19 17:30 attention-ckpt-11000.h5
// etc
-rw-rw-r-- 1 seita seita 1.7M Jul 19 16:27 attention-ckpt-8000.h5
-rw-rw-r-- 1 seita seita 1.7M Jul 19 16:48 attention-ckpt-9000.h5
-rw-rw-r-- 1 seita seita 3.4M Jul 19 13:54 transport-ckpt-1000.h5
-rw-rw-r-- 1 seita seita 3.4M Jul 19 17:09 transport-ckpt-10000.h5
-rw-rw-r-- 1 seita seita 3.4M Jul 19 17:30 transport-ckpt-11000.h5
// etc
-rw-rw-r-- 1 seita seita 3.4M Jul 19 16:04 transport-ckpt-7000.h5
-rw-rw-r-- 1 seita seita 3.4M Jul 19 16:27 transport-ckpt-8000.h5
-rw-rw-r-- 1 seita seita 3.4M Jul 19 16:48 transport-ckpt-9000.h5
```

It should be straightforward to load the agents to do further testing if needed.

Upload results from pickle files and checkpoints (careful, checkpoints can take
a lot of space):

```
gsutil -m cp ~/pybullet-def-envs/*.pkl gs://research-brain-deformable-manipulation-xgcp/results/
gsutil -m cp -r ~/pybullet-def-envs/checkpoints/* gs://research-brain-deformable-manipulation-xgcp/checkpoints/
```

Download pickle results or checkpoints:

```
gsutil -m cp gs://research-brain-deformable-manipulation-xgcp/results/*  .
gsutil -m cp -r gs://research-brain-deformable-manipulation-xgcp/checkpoints/cloth* checkpoints/
```

## Tensorboard

Finally, we have results stored for Tensorboard in `logs/[agent-name]/[env-name]/[date]`.

```
seita@danielseita-experimental-01:~/pybullet-def-envs$ ls -lh logs/transporter/
total 12K
drwxr-xr-x 4 seita seita 4.0K Jul 17 20:14 aligning
drwxr-xr-x 4 seita seita 4.0K Jul 19 14:30 cloth-flat
drwxr-xr-x 5 seita seita 4.0K Jul 18 00:51 insertion
```


[1]:https://partner-code.googlesource.com/project-reach/+/refs/heads/master/experimental/ravens/
[2]:https://programmer.ink/think/ubuntu-installs-multiple-cuda-versions-and-can-switch-at-any-time.html
[3]:https://blender.stackexchange.com/questions/144083/how-to-get-blender-2-80-to-render-through-an-ssh-connection-minimal-working-ex/176110#176110
