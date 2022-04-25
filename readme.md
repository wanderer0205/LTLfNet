## directories instructions

- `genGraphData.py`: parse LTLf formulae as Abstract Syntactic Tree
- `layers.py`: implementation of network layers used by `model.py`
- `model.py`: LTLfNet implementation
- `train.py`: implementation of model training
- `AIJ-data`: industrial datasets that have been transformed to only contain ['&', '|', '!', 'F', 'G', 'X', 'U']
- `data`: train set, validation set and test set
  - LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-contrasive-[20,100]: contrastive datasets consist of LTLf formulae of size [20,100] with no more than 100 variables
  - LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-random-[20,100]: random datasets consist of LTLf formulae of size [20,100] with no more than 100 variables
  - LTLSATUNSAT-and-or-not-F-G-X-until-100-random: five random datasets, each consisting of LTLf formulae within different ranges of size with no more than 100 variables

## model training

(1) requirements: 

+ `pip install -r requirements.txt`
+ pytorch 1.8.1 or above: https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

(2) enter `LTLfNet` directory

(3) run the following command on shell to train and test LTLfNet with the contrastive dataset on the 0-th GPU

    python3 train.py --model LTLfNet --device 0 --ts --rd 1