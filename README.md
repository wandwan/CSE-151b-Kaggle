# CSE-151b-Kaggle
Kaggle Project with team for cse 151

Config:
Needs BOTH pip and conda environments, add more requirements here as they are added to the project.
Pip:
``` 
cd <path-to-project>
pip install -r requirements.txt 
```
Conda:
Install Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
Then run:
```
conda config --prepend channels conda-forge
conda create -n ox --strict-channel-priority osmnx
```
Before running the script, run:
```
conda activate ox
```
