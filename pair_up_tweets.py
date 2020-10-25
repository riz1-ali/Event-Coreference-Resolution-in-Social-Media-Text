import pickle
import tokenizer as tk

def gen_distance_vec(tweet, trigger_word):
    trig = trigger_word.split()
    trig = trig[len(trig) // 2]
    words = tk.tokenize(tweet)
    pos = -1
    for i in range(len(words)):
        if trig == words[i]:
            pos = i
            break
    return [i - pos for i in range(len(words))]


tweet_pairs = []
distance_vecs = []
file_path = './generated_dataset.txt'
pair_up_threshold = 604800000

with open(file_path, 'r') as f:
    tot_data = []
    for i in f:
        data = i.strip('\n').split('\t')
        data[-1] = data[-1].lower()
        distance_vecs.append(gen_distance_vec(data[-1], data[3]))
        tot_data.append(data)
    for i in range(len(tot_data)):
        for j in range(len(tot_data)):
            if abs(int(tot_data[i][4]) -
                   int(tot_data[j][4])) <= pair_up_threshold:
                tweet_pairs.append([i, j])

minv = 200
for i in distance_vecs:
    minv = min(minv,min(i))

minv -= 1
for i in range(len(distance_vecs)):
    for j in range(len(distance_vecs[i])):
        distance_vecs[i][j] -= minv

with open('distance_vectors.pkl', 'wb') as f:
    pickle.dump(distance_vecs, f)
with open('tweet_pairs.pkl', 'wb') as f:
    pickle.dump(tweet_pairs, f)
