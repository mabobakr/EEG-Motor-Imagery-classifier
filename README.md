# EEG-Motor-Imagery-classifier
A machine learning model to classify EEG motor imagery tasks



# Note
You should have a folder named "data" in the repo root directory containing the gdf data files.


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

### Run the model simulation

<b>running using gitbash</b>

<b>running on linux</b>
```bash
export FLASK_APP=flaskr
export FLASK_DEBUG=1
flask run --reload --host=0.0.0.0

```
### Run the GUI

`python -m flaskr`
