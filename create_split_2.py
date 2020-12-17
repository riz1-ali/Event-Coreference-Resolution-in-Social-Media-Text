import pickle
import tokenizer as tk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataloader import dataset, collate_fn
from torch.utils import data


def gen_distance_vec(tweet, trigger_word):
    trig = tk.tokenize(trigger_word)
    words = tk.tokenize(tweet)
    pos = -1
    for i in range(len(words)):
        if trig[len(trig) // 2] == words[i]:
            pos = i
            break
    first_p, last_p = -1, -1
    for i in range(len(words)):
        if trig[0] == words[i]:
            first_p = i
            break
    if first_p == -1:
        for i in range(len(words)):
            if trig[0] in words[i] or words[i] in trig[0]:
                first_p = i
                break
    for i in range(len(words) - 1, -1, -1):
        if trig[-1] == words[i]:
            last_p = i
            break
    if last_p == -1:
        for i in range(len(words)):
            if trig[-1] in words[i] or words[i] in trig[-1]:
                last_p = i
                break
    return [i - pos for i in range(len(words))], [first_p, last_p]


def common_words(tweet1, tweet2):
    tweet_1 = tk.tokenize(tweet1)
    tweet_2 = tk.tokenize(tweet2)
    counts = 0
    for i in tweet_1:
        if i in tweet_2:
            counts += 1
    return counts


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


def gen_data(tweet_data, tag):
    tweet_pairs = []
    distance_vecs = []
    trigger_word_pos = []
    first_last_pos = []
    labels_data = []
    common_words_data = []
    day_difference_data = []
    distance_vector_data = []
    pair_up_threshold = 604800000

    tot_data = []
    for data in tweet_data:
        dis_v, pos = gen_distance_vec(data[-1], data[3])
        distance_vecs.append(dis_v)
        first_last_pos.append(pos)
        tot_data.append(data)
    
    minv = 200
    for i in distance_vecs:
        minv = min(minv, min(i))
    minv -= 1
    for i in range(len(distance_vecs)):
        for j in range(len(distance_vecs[i])):
            distance_vecs[i][j] -= minv
    
    for i in tqdm(range(len(tot_data))):
        for j in range(i + 1, len(tot_data)):
            if int(tot_data[j][4]) - int(tot_data[i][4]) <= pair_up_threshold:
                label = 0
                if tot_data[i][1] == tot_data[j][1]:
                    label = 1
                tweet_pairs.append([tot_data[i][-1], tot_data[j][-1]])
                labels_data.append(label)
                trigger_word_pos.append([first_last_pos[i],first_last_pos[j]])
                distance_vector_data.append([distance_vecs[i],distance_vecs[j]])
                common_words_data.append(common_words(
                    tot_data[i][-1], tot_data[j][-1]))
                day_difference_data.append(
                    (int(tot_data[j][4]) - int(tot_data[i][4])) / (1000 * 60 * 60 * 24))
    save_loader(tweet_pairs, distance_vector_data, trigger_word_pos,
                labels_data, common_words_data, day_difference_data, tag)


data_ = []
file_path = './FinalDataset.csv'
with open(file_path, 'r') as f:
    for i in f:
        j = i.strip('\n').split('\t')
        j[-1] = j[-1].lower()
        data_.append(j)

indices = list(range(len(data_)))
train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

train_data, val_data, test_data = [], [], []

for i in train_idx:
    train_data.append(data_[i])
for i in test_idx:
    test_data.append(data_[i])
for i in val_idx:
    val_data.append(data_[i])

gen_data(train_data, "train")
gen_data(test_data, "test")
gen_data(val_data, "val")
