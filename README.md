This code implements the CMDD. We use Python 3.9 to implement the proposed method, and all experiments are run on the machine of linux system. 
To run the model, you need to put the dataset file in the Multivariate_ts, set the necessary hyperparameters, and run the python file train.py.

# LONG-TAILED TIME SERIES CLASSIFICATION WITH NOISY LABELS

The source code is for reproducing experiments of the paper entitled "LONG-TAILED TIME SERIES CLASSIFICATION WITH NOISY LABELS"

# Datasets
The UCR dataset used in the paper are available at : http://www.timeseriesclassification.com/ .

# Usage

## Install packages
You can  create a new environment of python 3.9  and use the packages listed in requirements.txt
`pip install -r requirements.txt`
pytorch install
`pip install torch==1.10.2+cu113 torchaudio==0.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`


## Run
`python val.py --noise 0.1 --dataset PenDigits --epoch 500 --batch-size 1024`

# Supplement
All modules have been uploaded, and the complete training file will be uploaded after this paper is accepted for publication.




