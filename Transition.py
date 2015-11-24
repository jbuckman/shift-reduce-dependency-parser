#!/usr/bin/env python

class Transition:
	Shift=0
	LeftArc=1
	RightArc=2

	ID=0
	ALL_LABELS =('abbrev acomp advcl advmod amod appos ' +  \
			'attr aux auxpass cc ccomp complm conj cop csubj ' + \
			'dep det dobj expl infmod iobj mark measure neg ' +  \
			'nn nsubj nsubjpass null num number parataxis partmod ' + \
			'pcomp pobj poss possessive preconj pred predet ' +  \
			'prep prt punct purpcl quantmod rcmod rel tmod ' +   \
			'xcomp').split()
	ALL_LABEL_ARGS = [None] + ALL_LABELS
	ALL_LABEL_IDXS = {v:i for i,v in enumerate(ALL_LABEL_ARGS)}

	def __init__(self, transitionType, label=None, score=0):
		self.transitionType = transitionType
		self.label = label
		if not label in Transition.ALL_LABEL_ARGS:
			raise Exception("WHERE DID THE LABEL %s COME FROM" % (label))
		self.score = score
		self.ID = Transition.ID
		Transition.ID += 1

	def __str__(self):
		return 'Transition of type %d with label %s' % (self.transitionType, self.label)

	def __repr__(self):
		return str(self)

	def __eq__(self, other):
		return self.transitionType == other.transitionType \
		and self.label == other.label

	def __ne__(self, other):
		return not (self == other)

	def __hash__(self):
		return self.ID
		
	def to_category(self):
		return int(self.transitionType)*len(Transition.ALL_LABEL_ARGS) + Transition.ALL_LABEL_IDXS[self.label]
	
	@staticmethod
	def from_category(catNum, score=0):
		transitionType = int(catNum/len(Transition.ALL_LABEL_ARGS))
		label = Transition.ALL_LABEL_ARGS[int(catNum%len(Transition.ALL_LABEL_ARGS))]
		return Transition(transitionType, label, score)