Event Coreference Resolution in Social Media Text
============================

## Prerequisites
- [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)


## To set up the repo.
```
git clone https://github.com/riz1-ali/Event-Coreference-Resolution-in-Social-Media-Text/
cd Event-Coreference-Resolution-in-Social-Media-Text
conda create -n EventCoref python=3.7
conda activate EventCoref
pip install -r requirements.txt
conda install -c pytorch pytorch
```

## Running Code
To fetch and preprocess the dataset, please checkout the ```fetch_dataset.ipynb``` notebook. However, the tweets are saved in ```generated_dataset.txt```. 

The preprocessed tweets are then saved in ```FinalDataset.csv```.

Once this is done run the following commands
```
python pair_up_tweets.py
```
This will generate the actual dataset with labels and pairs of tweets. This is then read straight into the dataloader. Setting up the dataloader is specified in ```dataloader.py``` which is then later used in ```Model.ipynb```.