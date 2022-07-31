# EEG-Motor-Imagery-classifier
A machine learning model to classify EEG motor imagery tasks of BCI IV 2a dataset



# Prerequistes
You should have a folder named "data" containing the BCI IV 2a dataset GDF files in the repo root directory.


## Getting Started

### Create a Virutal Enviornment

Follow instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to create and activate virtual enviornment for this project.

to create virtual environment run
```bash
python -m venv env
```

once the virtual environment created run this command
```bash
source env/bin/activate

```
### Install Dependencies

Run `pip install -r requirements.txt` to install any dependencies.


## How to run

Once you have installed all requirements, do the following:

1- In model.ipynb, uncomment the *convert_data* function call and run all cells.

2- In model.ipynb, if you have already run the function call of *convert_data* then you don't need to do it again.

