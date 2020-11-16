import pickle
from sklearn.model_selection import train_test_split
from dataloader import dataset, collate_fn
from torch.utils import data

data_ = []
file_path = './FinalDataset.csv'
with open(file_path, 'r') as f:
    for i in f:
        j = i.strip('\n').split('\t')
        data_.append(j[-1].lower())

tweet_pairs, distance_vectors = [], []

with open('./tweet_pairs.pkl', 'rb') as f:
    tweet_pairs = pickle.load(f)

with open('./distance_vectors.pkl', 'rb') as f:
    distance_vectors = pickle.load(f)

with open("./trigger_word_pos.pkl", 'rb') as f:
    trigger_word_pos = pickle.load(f)

indices = list(range(len(tweet_pairs)))


def save_loader(
        tweet_pair_data,
        distance_vector_data,
        trigger_word_pos_data,
        labels_data,
        common_words_data,
        day_difference_data,
        type):
    """
    Creates dataset and Dataloader objects from provided data and stores in pickle file
    """
    dataset_ = dataset(
        tweet_pair_data,
        distance_vector_data,
        trigger_word_pos_data,
        common_words_data,
        day_difference_data,
        labels_data)
    loader = data.DataLoader(
        dataset_,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True)
    with open(f"{type}_loader.pkl", "wb") as f:
        pickle.dump(loader, f)


X_train, X_test, y_train, y_test = train_test_split(
    indices, indices, test_size=0.6, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.375, random_state=42)

X_train = [tweet_pairs[i] for i in X_train]
X_val = [tweet_pairs[i] for i in X_val]
X_test = [tweet_pairs[i] for i in X_test]


tweet_pair_data_train = [[data_[i[0]], data_[i[1]]] for i in X_train]
distance_vector_data_train = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_train]
trigger_word_pos_data_train = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_train]
labels_data_train = [i[2] for i in X_train]
common_words_data_train = [i[3] for i in X_train]
day_difference_data_train = [i[4] for i in X_train]


tweet_pair_data_val = [[data_[i[0]], data_[i[1]]] for i in X_val]
distance_vector_data_val = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_val]
trigger_word_pos_data_val = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_val]
labels_data_val = [i[2] for i in X_val]
common_words_data_val = [i[3] for i in X_val]
day_difference_data_val = [i[4] for i in X_val]


tweet_pair_data_test = [[data_[i[0]], data_[i[1]]] for i in X_test]
distance_vector_data_test = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_test]
trigger_word_pos_data_test = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_test]
labels_data_test = [i[2] for i in X_test]
common_words_data_test = [i[3] for i in X_test]
day_difference_data_test = [i[4] for i in X_test]

save_loader(
    tweet_pair_data_train,
    distance_vector_data_train,
    trigger_word_pos_data_train,
    labels_data_train,
    common_words_data_train,
    day_difference_data_train,
    "train")

save_loader(
    tweet_pair_data_test,
    distance_vector_data_test,
    trigger_word_pos_data_test,
    labels_data_test,
    common_words_data_test,
    day_difference_data_test,
    "test")

save_loader(
    tweet_pair_data_val,
    distance_vector_data_val,
    trigger_word_pos_data_val,
    labels_data_val,
    common_words_data_val,
    day_difference_data_val,
    "val")
