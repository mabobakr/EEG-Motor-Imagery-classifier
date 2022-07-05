# EEG-Motor-Imagery-classifier
A machine learning model to classify EEG motor imagery tasks



# Note
You should have a folder named "data" in the repo root directory containing the gdf data files.

# Flask Recap

A simple flask server to demonstrate basic flask.

## Getting Started

### Create a Virutal Enviornment

Follow instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to create and activate virtual enviornment for this project.

### Install Dependencies

Run `pip install -r requirements.txt` to install any dependencies.

### Run the Server

<b>running using gitbash</b>


<b>running on linux</b>
```bash
source env/bin/activate
export FLASK_APP=flaskr
export FLASK_DEBUG=1
flask run --reload --host=0.0.0.0

```
### Run the GUI

`python -m flaskr`
