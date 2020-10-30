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

### Preprocessing
To fetch and preprocess the dataset, please checkout the ```fetch_dataset.ipynb``` notebook. However, the tweets are saved in ```generated_dataset.txt```. 

The preprocessed tweets are then saved in ```FinalDataset.csv```.

Once this is done run the following commands
```
python pair_up_tweets.py
```
This will generate the actual dataset with labels and pairs of tweets. This is then read straight into the dataloader. Setting up the dataloader is specified in ```dataloader.py``` which is then later used in ```Model.ipynb```.

### Creating Dataloaders
After pairing up the tweets, the next thing is to split the data into train, test and validation set which is done in the ```create_split.py``` file. 
```
python create_split.py
```
The size of the splits can be changed in lines 52 and 55 of the file. This script will create pickle files storing the train, validation and test dataloader as ```train_loader.pkl```, ```val_loader.pkl``` and ```test_loader.pkl``` respectively. These dataloaders can be loaded for direct use into the model.

### Training the Model
The model classes are implemented in the ```models.py``` file. The utility functions for testing and validation are implemented in ```utils.py```.

The driver code for training is in ```cleanModel.py```. The logging of metrics is done in [wandb](https://wandb.ai/manan_goel/EventCoreference?workspace=user-). Click on this link to check. Hyperparameters to the model can be changed using command line arguments. To see possible hyperparameters, run
```
python cleanModel.py --help
```
As of now the only models supported are bidirectional ```LSTM``` and ```GRU``` along with the scheduler supported is ReduceLROnPlateau.

To train the model without wandb logging, please run
```
python cleanModel.py --modelType LSTM --lr 1e-2 --device GPU --hidden_size 64 --num_epochs 50 --use_scheduler True
```
To train the model with wandb logging, please run
```
python cleanModel.py --modelType LSTM --lr 1e-2 --device GPU --hidden_size 64 --num_epochs 50 --use_scheduler True --use_wandb True
```
