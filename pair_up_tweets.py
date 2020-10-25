import pickle
import tokenizer as tk

def gen_distance_vec(tweet, trigger_word):
	trig = tk.tokenize(trigger_word)
	words = tk.tokenize(tweet)
	pos = -1
	for i in range(len(words)):
		if trig[len(trig)//2] == words[i]:
			pos = i
			break
	first_p,last_p = -1,-1
	for i in range(len(words)):
		if trig[0] == words[i]:
			first_p = i
			break
	if first_p==-1:
		for i in range(len(words)):
			if trig[0] in words[i] or words[i] in trig[0]:
				first_p = i
				break
	for i in range(len(words)-1,-1,-1):
		if trig[-1] == words[i]:
			last_p = i
			break
	if last_p==-1:
		for i in range(len(words)):
			if trig[-1] in words[i] or words[i] in trig[-1]:
				last_p = i
				break
	return [i - pos for i in range(len(words))],[first_p,last_p]


tweet_pairs = []
distance_vecs = []
first_last_pos = []
file_path = './generated_dataset.txt'
pair_up_threshold = 604800000

with open(file_path, 'r') as f:
	tot_data = []
	for i in f:
		data = i.strip('\n').split('\t')
		data[-1] = data[-1].lower()
		dis_v,pos = gen_distance_vec(data[-1].lower(), data[-3].lower())
		distance_vecs.append(dis_v)
		first_last_pos.append(pos)
		tot_data.append(data)
	for i in range(len(tot_data)):
		for j in range(i+1,len(tot_data)):
			if int(tot_data[j][-2]) - int(tot_data[i][-2]) <= pair_up_threshold:
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
with open('trigger_word_pos.pkl','wb') as f:
	pickle.dump(first_last_pos,f)