def gen_distance_vec(tweet,trigger_word):
	trig = trigger_word.split()
	trig = trig[len(trig)//2]
	words = tweet.split()
	pos = -1
	for i in range(len(words)):
		if trig == words[i]:
			pos = i
			break
	return [i-pos for i in range(len(words))]

tweet_pairs = []
distance_vecs = []
file_path = './generated_dataset.txt'
pair_up_threshold = 604800000
with open(file_path,'r') as f:
	line = 0
	tot_data = []
	for i in f:
		data = i.strip('\n').split('\t')
		distance_vecs.append(gen_distance_vec(data[-1],data[3]))
		tot_data.append(data)
		line += 1
	for i in range(len(tot_data)):
		for j in range(len(tot_data)):
			if abs(int(tot_data[i][4])-int(tot_data[j][4])) <= pair_up_threshold:
				tweet_pairs.append([i,j])

f = open('distance_vectors.txt','w')
f.write(str(distance_vecs))
f.close()
f = open('tweet_pairs.txt','w')
f.write(str(tweet_pairs))
f.close()