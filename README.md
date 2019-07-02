# Asymptotic Risk of Bézier Simplex Fitting
This project provides source files for reproducing the above paper submitted to NeurIPS 2019 and experiments therein.

## Requirements
- Ubuntu 16.04 LTS or above
- TexLive full 2018 or above
- GNU Make 4.2.0 or above
- Python 3.6.0 or above
- Packages for uses: [requirements.txt](requirements.txt)
- Packages for developers: [requirements-dev.txt](requirements-dev.txt)


## How to reproduce our results
`Dockerfile` is provided for the required software pre-installed.

First, build an image on your machine.

```
$ git clone https://github.com/rafcc/neurips-19.6340.git
$ cd neurips-19.6340
$ docker build --build-arg http_proxy=$http_proxy -t .
```

Then, run a container

```
$ docker run --rm -v $(pwd):/data -it rafcc/aaai-19
```

In the container, the experimental results can be reproduced by simply run the following commands:

```
$ cd src
$ python exp_synthetic_instances_D2.py
$ python exp_synthetic_instances_D3.py
$ python exp_practical_instances.py
```

When you run the following commands, then, you get the following directories which include experimental results:

```
../results_synthetic_instances_borges_inductive_optimal/
../results_synthetic_instances_inductive_nonoptimal/
../results_practical_instances/
```
