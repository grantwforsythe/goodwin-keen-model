## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
Using a Goodwin and Goodwin-Keen model to model a macroeconomy. This repo was created for a group project (Romi Lifshitz, Arthur M\'endez-Rosales, Sara Saad, Grant Forsythe, Gheeda Mourtada, and Jacob Ronen Keffer) for Math 3MB3 (McMaster). Checkout our final report and presentation under the docs directory.

## Technologies
Project is created with:
* Python version: 3.8
	
## Setup
It is a best practice to run program files in a virtual environment as
it allows the program to run with it's own separate dependencies.

To initialize a virtual environment use the command:
```console
cd path_to_project/
python -m venv .venv
```
To activate the environment:

Mac/Linux:
```console
source my_env/bin/activate
```
Windows:
```console
.\venv\Scripts\activate
```
To verify that your virtual environment is activated, your command line
should look like the following.
```console
(.venv) C:\Users\name\Desktop\Repos\math3mb3>
```
Then, to install the required dependices, run the following command.
```console
pip install -r requirements.txt
```
To deactivate simply use the command:
```console
deactivate
```
