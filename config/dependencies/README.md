# Conda Environment

To reinstall the conda environemnt running this command from this folder:

```bash
conda env create -f environment.yml
```

# Virtualenv Environment
Make sure to have the virtualenv environment installed, if not run this command:
```bash
pip install virtualenv
```
Follow this steps to create the virtualenv environment and setup the correct dependencies:
1. go to the root folder of this repository
1. run `virtualenv venv` to create your new environment (in the folder `venv`)
1. run `source venv/bin/activate` to enter the virtual environment
1. run `pip install -r venv_requirements.txt` to install the requirements in the current environment


