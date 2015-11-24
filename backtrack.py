from copy import copy
from collections import defaultdict
from itertools import product

# A choice is a tuple (score, (function, args))

class DecisionNode(object):
	ALL = []
	ID = 0
	
	def __init__(self, parent):
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
		
	def __str__(self):
		return '.'+str(self.ID)+'.'
	def __repr__(self):
		return str(self)
		
	def explore(self, choices):
		if not self.frontier:
			raise Exception('expanded something not on frontier:'+str(self))
		for item in choices:
			new_node = DecisionNode(self)
			self.children[new_node] = item[1]
			self.choices[item[1]] = new_node
		self.sorted_choices = sorted(choices, reverse=True)
		self.frontier = False
		self.current_i = 0
		self.calculate_score_diff()
		
	def calculate_score_diff(self):
		if len(self.sorted_choices) > self.current_i + 1:
			self.score_diff = self.sorted_choices[self.current_i][0] - self.sorted_choices[self.current_i+1][0]
		else:
			self.score_diff = 999999999999
			self.ongoing = False
	
	def get_next_func(self):
		ans = self.sorted_choices[self.current_i][1]
		return ans

def get_all_choices(stack, stream):
	choices = []
	if stream:
		choices.append((matchsr.SHIFT_SCORE, (applysr.shift, ())))
	for function in matchsr.matchfns:
		choices += function(stack, stream)
	return choices

def parse(sentence, max_iterations=-1):
	context = Indv(parent="universe")
	context.activate_context()
	
	words = sentence.split(" ")
	DecisionNode.ALL = []
	
	stack = []
	stream = copy(words)
	
	base = DecisionNode(None)
	choices = get_all_choices(stack, stream)
	base.explore(choices)
	current_interpretation = None
	
	# when a score is penalized, all of its children are too
	score_penalty = 0
	
	nnn = 0
	while [node for node in DecisionNode.ALL if node.ongoing]:
		if nnn > max_iterations and max_iterations != -1:
			raise Exception('no parse found')
			
		print '\nNEW INTERPRETATION', nnn
		nnn += 1
		### this loop happens once per interpretation ###
		stack = []
		stream = copy(words)
		# activate a context
		if current_interpretation:
			current_interpretation.deactivate_context()
			current_interpretation.is_a("incorrect interpretation")
		current_interpretation = NewInterpretation()
		current_interpretation.activate_context()
			
		# re-create the situation of the base
		tracer = base
		path = [tracer]
		while tracer.parent != None:
			tracer = tracer.parent
			path.append(tracer)		
		path.reverse()
				
		for node_i in range(len(path)-1):
			#~ print 'building base:', stack, stream
			func, args = path[node_i].children[path[node_i+1]]
			func(stack, stream, *args)
		#~ print 'starting base:', base, stack, stream
		ancestor_penalty = base.ancestor_penalty
		
		# take the new best choice
		func, args = base.get_next_func()
		func(stack, stream, *args)
		#~ print '  new choice:', str(base.current_i+1)+'/'+str(len(base.sorted_choices)), func.__name__, args
		base = base.choices[(func, args)]
		#~ print 'resulting base:', base, stack, stream
		
		# from the new place, push forward by exploring and taking the best choice until you have no choices
		choices = get_all_choices(stack, stream)
		while choices:
			base.ancestor_penalty += ancestor_penalty
			base.explore(choices)
			func, args = base.get_next_func()
			func(stack, stream, *args)
			#~ print ' ', str(base.current_i+1)+'/'+str(len(base.sorted_choices)), func.__name__, args			
			base = base.choices[(func, args)]
			#~ print 'resulting base:', base, stack, stream
			choices = get_all_choices(stack, stream)
		base.explore(choices)

		# when we have no choices, check whether we have a success:
		if not stream and len(stack) == 1:
		#	if so, break and return the final SemanticNode off of the stack
			current_interpretation.is_a("correct interpretation")
			return stack[0]
			
		else:
		#	if not, increment current_i of parent of failnode, and use score_diffs to choose new base
			try:
				print nnn, stack, stream
				score, base = min([(node.score_diff+node.ancestor_penalty, node) for node in DecisionNode.ALL if node.ongoing and not node.frontier])
				base.current_i += 1
				base.ancestor_penalty += base.score_diff
				base.calculate_score_diff()
				#~ raw_input()
				
			except:
				raise Exception('no nodes left to try')
	raise Exception('no parse found')
