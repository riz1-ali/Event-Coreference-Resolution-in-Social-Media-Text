import pickle
import tokenizer as tk
from tqdm import tqdm


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


tweet_pairs = []
distance_vecs = []
first_last_pos = []
file_path = './FinalDataset.csv'
pair_up_threshold = 604800000

with open(file_path, 'r') as f:
    tot_data = []
    for i in f:
        data = i.strip('\n').split('\t')
        data[-1] = data[-1].lower()
        dis_v, pos = gen_distance_vec(data[-1].lower(), data[3].lower())
        distance_vecs.append(dis_v)
        first_last_pos.append(pos)
        tot_data.append(data)
    for i in tqdm(range(len(tot_data))):
        for j in range(i + 1, len(tot_data)):
            if int(tot_data[j][4]) - int(tot_data[i][4]) <= pair_up_threshold:
                label = 0
                if tot_data[i][1] == tot_data[j][1]:
                    label = 1
                tweet_pairs.append([i, j, label, common_words(tot_data[i][-1], tot_data[j][-1]),
                                    (int(tot_data[j][4]) - int(tot_data[i][4])) / (1000 * 60 * 60 * 24)])

minv = 200
for i in distance_vecs:
    minv = min(minv, min(i))

minv -= 1
for i in range(len(distance_vecs)):
    for j in range(len(distance_vecs[i])):
        distance_vecs[i][j] -= minv

with open('distance_vectors.pkl', 'wb') as f:
    pickle.dump(distance_vecs, f)
with open('tweet_pairs.pkl', 'wb') as f:
    pickle.dump(tweet_pairs, f)
with open('trigger_word_pos.pkl', 'wb') as f:
    pickle.dump(first_last_pos, f)
