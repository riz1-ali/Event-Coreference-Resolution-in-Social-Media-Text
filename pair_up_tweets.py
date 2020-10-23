def gen_distance_vec(tweet,trigger_word):
	pos = -1
	t = tweet.split()
	for i in range(len(t)):
		if trigger_word == t[i]:
			pos = i
			break
	return [i-pos for i in range(len(t))]

tweet_pairs = []
distance_vecs = []
file_path = './generated_dataset.txt'
with open(file_path,'r') as f:
	line = 0
	trigger_word = {}
	for i in f:
		data = i.strip('\n').split('\t')
		print(data[3])
		distance_vecs.append(gen_distance_vec(data[-1],data[3]))
		if data[3] not in trigger_word.keys():
			trigger_word[data[3]] = []
		trigger_word[data[3]].append(line)
		line += 1
	for word in trigger_word.keys():
		if len(trigger_word[word]) > 1:
			for i in range(len(trigger_word[word])):
				for j in range(i+1,len(trigger_word[word])):
					tweet_pairs.append([trigger_word[word][i],trigger_word[word][j]])
# for i in tweet_pairs:
# 	print(i)