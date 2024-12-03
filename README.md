
[![Code-quality](https://github.com/goyalpike/quadratic-stable-opinf/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/goyalpike/quadratic-stable-opinf/actions/workflows/pre-commit.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# 🎯 **[Guaranteed Stable Quadratic Models and Their Applications in SINDy and Operator Inference](https://arxiv.org/abs/2308.13819)**

This repository contains the Python implementation using the PyTorch framework to learn quadratic stable systems. It is based on the results presented in [1] and has particularly applications in operator inference and SINDy approaches. Furthermore, we have blended the fourth-order Runge-Kutta scheme; thus, the method does not require the computation of derivative information to learn dynamical models. Hence, it holds a key advantage when data are corrupted and sparsely sampled. 

## 🌟 **Methodology Highlights**

1. Collect measurement data  
2. Utilize parameterization for suitable stable quadratic dynamical systems
3. Set-up an optimization problem by incorportating the fourth-order Runge-Kutta scheme
4. Solve the optimization problem using gradient-decent within the spirit of NeuralODEs 
 	
For high-dimensional data, we utilize proper orthogonal decoposition to obtain a low-dimensional representation. To solve the underlying optimization problem, we utilize automatic differentiation implemented in PyTorch. 


## 📦 **Setup Instructions**
For reproducibility, we suggest to create a conda virtual enviornment that installs required python and pytorch verions. The reason is to have automatically cross plateform installation of pytorch since with poetry is rather complicated. It is followed by installing dependencies using poetry. The instrusions are as follows:

```bash
conda env create -f environment.yml
conda activate quad-stable-opinf
poetry install --all-extras
```

If you are using headless server, then run 

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

so that poetry does not hang. 


## 📁 **Examples**
 Numeical exampels considered in the paper are in `Example folder`. All the results reported in the paper can be reproduced using `run_examples.sh`. 

## Docker Setup
The results can be reproduced via docker image as well. Please first build a docker image using e.g.,
```bash
docker build -t qs-opinf-image-cpu .
```
Then, run the following command to create a container 
```bash
docker run -v LocalPath_to_Save_Results:/app/Examples/Results -it qs-opinf-image-cpu
```
In the above, we have mounted the volumn so that the generated resulted can be saved locally, otherwise they will be gone once containers is removed. Once the container is up and runnning, you can generate results using 
```bash
bash run_examples.sh
```

Note that here docker image is generated for `cpu` only. However, if `gpu` compatible installation is needed then remove `cpuonly` dependency from `environment.yml`.

## 📜 **License**
See the [LICENSE](LICENSE) file for license rights and limitations (MIT).



## 📖 **Reference**
[1]. P. Goyal, I. Pontes Duff and P. Benner, [Guaranteed Stable Quadratic Models and their applications in SINDy and Operator Inference](https://arxiv.org/abs/2308.13819), arXiv:2308.13819, 2023.
<details><summary>📚 BibTeX</summary><pre>
@TechReport{morGoyPB23,
  author =       {Goyal, P., Pontes Duff, I., and Benner, P.},
  title =        {Guaranteed Stable Quadratic Models and their applications in SINDy and Operator Inference},
  institution =  {arXiv},
  year =         2023,
  type =         {e-print},
  number =       {2308.13819},
  url =          {https://arxiv.org/abs/2308.13819},
}
</pre></details>

## 📬 **Contact**
For any further query, kindly contact [Pawan Goyal](mailto:goyalp@mpi-magdeburg.mpg.de). 