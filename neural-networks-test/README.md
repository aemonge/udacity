# Installation and revision method

## Part 1 - Part 1 - Development Notebook

To see part one of this exam, please install Jupiter and open the notebook.

      pip install jupyter
      jupyter notebook ./Image Classifier Project.ipynb

## Part 2  - Command Line Application

`cd ./commandline`

To install and test the command line application, please install [conda](https://docs.anaconda.com/anaconda/install/index.html) as it's a environment manager and will later ease the installation.


After that, simply create the environment by:

      conda env create -f environment.yml -n aemonge-udacity
      conda activate aemonge-udacity

### Run

And start testing the script by either using the wrapper:

      ./main --help

Or invoking individually the scripts:

      python ./cli/train.py --help
      python ./cli/test.py --help
      python ./cli/sanity.py --help
      python ./cli/predict.py --help

🎔
Happy revision!

-- aemonge
