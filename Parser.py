import sys
import time
import cPickle
from Oracle import Oracle
from Models import PerceptronModel, SVCModel, RandomForestModel, SKPerceptronModel
from Transition import Transition
from copy import copy
from operator import add, mul

class Parser:	
	def __init__(self, labeled):
		self.labeled = labeled

	def initialize(self, sentence):
		self.root = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '-1', 'ROOT', 'ROOT', 'ROOT']
		self.buff = [self.root] + list(reversed(sentence))
		self.stack = list()
		self.arcs = {}
		self.labels = {}
		self.transitions = list()
		self.leftmostChildren = {node[0]:{} for node in self.buff}
		self.rightmostChildren = {node[0]:{} for node in self.buff}
		for node in sentence:
			if int(node[0]) < int(node[6]) or int(node[6]) == 0:
				self.leftmostChildren[node[6]][node[0]] = node[7]
			elif int(node[0]) > int(node[6]):
				self.rightmostChildren[node[6]][node[0]] = node[7]
		DecisionNode.ALL = []
		
	def execute_transition(self, transition):
		"""This function should take a transition object and apply to the current parser
		state. It need not return anything."""
		if transition.transitionType == Transition.Shift:
			self.stack.append(self.buff.pop(-1))
		elif transition.transitionType == Transition.LeftArc:
			self.arcs[self.stack[-2][0]] = self.stack[-1][0]
			self.labels[self.stack[-2][0]] = transition.label
			self.stack.pop(-2)
		elif transition.transitionType == Transition.RightArc:
			self.arcs[self.stack[-1][0]] = self.stack[-2][0]
			self.labels[self.stack[-1][0]] = transition.label			
			self.stack.pop(-1)

		self.transitions.append(transition)

	@staticmethod
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

	def output(self, sentence):
		for token in sentence:
			head = self.arcs.get(token[0], '0')
			label = self.labels.get(token[0], '_')
			label = label if label is not None else '_'
			token[6] = head
			token[7] = label
			print '\t'.join(token)
		print


	def train(self, trainingSet, model):
		corpus = Parser.load_corpus(trainingSet)	
		oracle = Oracle()
		i = 0
		for sentence in corpus:
			if i % 25 == 0:
				print >>sys.stderr, "%d/1921" % i
			i += 1
			self.initialize(sentence)
			while len(self.buff) > 0 or len(self.stack) > 1:
				transition = oracle.getTransition(self.stack, self.buff, \
					self.leftmostChildren, self.rightmostChildren, \
					self.arcs, self.labeled)
				model.learn(transition, self.stack, self.buff, \
					self.arcs, self.labels, self.transitions)
				self.execute_transition(transition)
		model.build_model()

	def parse(self, testSet, model, maxBacktracks):
		corpus = Parser.load_corpus(testSet)
		nnn = 0
		for sentence in corpus:
			print >>sys.stderr, "%d/71" % nnn
			DecisionNode.initialize()
			nnn += 1
			completed_parses = []
			self.initialize(sentence)
			base = DecisionNode(None, self.stack, self.buff, self.labels, self.transitions, self.arcs)
			attempts = 0
			score_penalty = 0
			first = None
			while attempts < maxBacktracks:
				attempts += 1
				
				# re-create the situation of the base
				self.stack = copy(base.stack)
				self.buff = copy(base.buff)
				self.labels = copy(base.labels)
				self.transitions = copy(base.transitions)
				self.arcs = copy(base.arcs)
				ancestor_penalty = base.ancestor_penalty
				
				# go at least one step further
				if base.frontier:
					choices = model.predict_all(self.stack, self.buff, self.arcs, self.labels, self.transitions)
					base.explore(choices)
				transition = base.get_next_transition()
				self.execute_transition(transition)
				base = DecisionNode(base, self.stack, self.buff, self.labels, self.transitions, self.arcs)
				choices = model.predict_all(self.stack, self.buff, self.arcs, self.labels, self.transitions)
					
				# push forward by exploring and taking the best choice until you have no choices
				while choices:
					base.ancestor_penalty += ancestor_penalty
					base.explore(choices)
					transition = base.get_next_transition()
					self.execute_transition(transition)
					base = DecisionNode(base, self.stack, self.buff, self.labels, self.transitions, self.arcs)
					choices = model.predict_all(self.stack, self.buff, self.arcs, self.labels, self.transitions)
				base.explore(choices)
				
				# confirm that the completed parse finished parsing the sentence
				if len(self.buff) > 0 or len(self.stack) > 1:
					raise Exception("something broke")
				
				# save off the node
				completed_parses.append(base)
				if not first:
					first = base
				#~ print "parse finished with score", sum([t.score for t in base.transitions])
				
				# parse the next chain
				score, base = min([(node.score_diff+node.ancestor_penalty, node) for node in DecisionNode.ALL if node.ongoing and not node.frontier])
				base.current_i += 1
				base.ancestor_penalty += base.score_diff
				base.calculate_score_diff()
				
			completed_parses.sort(key=lambda x:reduce(add, [t.score for t in x.transitions]), reverse=True)
			# go to the state of the highest-scoring comlpleted decision node
			base = completed_parses[0]
			if base != first:
				print >>sys.stderr, "HAPPENED %s %s, diff %s" % (first, base, str(reduce(add, [t.score for t in base.transitions])-reduce(add, [t.score for t in first.transitions])))
			#~ print >>sys.stderr, str([(parse,sum([t.score for t in parse.transitions])) for parse in completed_parses[:10]])
			
			self.stack = base.stack
			self.buff = base.buff
			self.labels = base.labels
			self.transitions = base.transitions
			self.arcs = base.arcs
			
			self.output(sentence)

# DecisionNode objects form a trie used to guide the heuristic backtracking.
# A choice is a tuple (score, transition)
class DecisionNode(object):
	ALL = []
	ID = 0
	
	def __init__(self, parent, stack, buff, labels, transitions, arcs):
		DecisionNode.ALL.append(self)
		self.children = {}
		self.choices = {}
		self.sorted_choices = []
		self.current_i = 0
		self.parent = parent
		self.score_diff = 0
		self.ancestor_penalty = 0
		self.frontier = True
		self.ongoing = True
		self.ID = DecisionNode.ID
		DecisionNode.ID += 1
		
		self.update_state(stack, buff, labels, transitions, arcs)
		
	def __str__(self):
		return '.'+str(self.ID)+'.'
	def __repr__(self):
		return str(self)
	
	@staticmethod
	def initialize():
		DecisionNode.ALL = []
		DecisionNode.ID = 0
	
	def update_state(self, stack, buff, labels, transitions, arcs):
		self.stack = copy(stack)
		self.buff = copy(buff)
		self.labels = copy(labels)
		self.transitions = copy(transitions)
		self.arcs = copy(arcs)
		
	def explore(self, choices):
		if not self.frontier:
			raise Exception('expanded something not on frontier:'+str(self))
		#~ for item in choices:
			#~ new_node = DecisionNode(self)
			#~ self.children[new_node] = item[1]
			#~ self.choices[item[1]] = new_node
		self.sorted_choices = choices
		self.frontier = False
		self.current_i = 0
		self.calculate_score_diff()
		
	def calculate_score_diff(self):
		if len(self.sorted_choices) > self.current_i + 1:
			self.score_diff = self.sorted_choices[self.current_i][0] - self.sorted_choices[self.current_i+1][0]
		else:
			self.score_diff = 999999999999
			self.ongoing = False
	
	def get_next_transition(self):
		ans = self.sorted_choices[self.current_i][1]
		return ans

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--labeled', '-l', action='store_true')
	parser.add_argument('--save', '-s', action='store_true')
	parser.add_argument('--retrain', '-r', action='store_true')
	parser.add_argument('trainingcorpus', help='Training treebank')
	parser.add_argument('testset', help='Dev/test treebank')
	parser.add_argument('beamsize', help='Beam size for parse')
	args = parser.parse_args()

	if args.retrain:
		p = Parser(args.labeled)
		model = RandomForestModel(args.labeled)
		p.train(args.trainingcorpus, model)
		if args.save:
			cPickle.dump( p, open( "p2.pickle", "wb" ) )
			cPickle.dump( model, open( "model2.pickle", "wb" ) )
		p.parse(args.testset, model, int(args.beamsize))
	else:
		p = cPickle.load( open( "p2.pickle" ) )
		model = cPickle.load( open( "model2.pickle" ) )
		#~ with open("model1", "w") as f:
			#~ f.write("\n".join([str(item) for item in sorted(model.weights.items())]))	
		p.parse(args.testset, model, int(args.beamsize))
