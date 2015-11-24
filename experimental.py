import sys
from collections import defaultdict

def load_corpus(filename):
	print >>sys.stderr, 'Loading treebank from %s' % filename
	corpus = []
	sentence = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				corpus.append(sentence)
				sentence = []
			else:
				word = line.split('\t')
				sentence.append(word)
	print >>sys.stderr, 'Loaded %d sentences' % len(corpus)
	return corpus
	
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('trainingcorpus', help='Training treebank')
	args = parser.parse_args()
	corpus = load_corpus(args.trainingcorpus)
	
	left_constructions = defaultdict(lambda :defaultdict(int))
	right_constructions = defaultdict(lambda :defaultdict(int))
	vocab = defaultdict(lambda :defaultdict(int))
	
	for sentence in corpus:
		for word in sentence:
			id = word[0]
			key = word[7]
			left_construction = []
			right_construction = []
			for i, word2 in enumerate(sentence):
				if word2[0] == id:
					break
				if word2[6] == id:
					left_construction.append(word2[7])
			for word2 in sentence[i+1:]:
				if word2[6] == id:
					right_construction.append(word2[7])
			left_constructions[key][tuple(left_construction)] += 1
			right_constructions[key][tuple(right_construction)] += 1
			vocab[key][word[1].lower()] += 1
			
	for cons in sorted(right_constructions['null'], key=lambda x:(-right_constructions['null'][x], len(x), x)):
		print cons, right_constructions['null'][cons]
