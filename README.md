### Installation
- Run ```git clone https://github.com/griprox/naked-mole-rats-talk.git```
in the terminal or download and unzip repository manually (top right corner, Code-> Download ZIP).
- Install [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create and activate virtual environment using conda: \
   ```conda create -n envname python=3.7``` \
   ```conda activate envname```
- Install python modules and Jupyter Notebook by running \
```cd naked-mole-rats-talk```\
```pip install -r requirements.txt```\
```conda install notebook```\
```conda install nb_conda_kernels```
- Install CPU version of tensorflow 2.4.1 by running \
```pip install tensorflow==2.4.1```
- If you want to use GPU, follow the official [guidelines](https://www.tensorflow.org/install/pip)

Now, you need to ensure that paths in recordings_metadata.csv actually point
at the data on your machine. To achieve that, do the following:
- Save data folder side by side with the code, so that you project folder will look as follows:
``` 
|-- naked-mole-rats-talk
|   |-- src
|   |-- notebooks
|   |-- data
```
- Open terminal, and go to the project directory ```cd path/naked-mole-rats-talk/```
- Run [setup](/setup_paths.py) script: ``` python setup_paths.py data```.
- Open the recordings_metadata.csv file and confirm that paths there are correct.