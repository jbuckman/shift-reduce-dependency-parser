import sys
from Transition import Transition

class Oracle:
	def getTransition(self, stack, buff, leftmostChildren, rightmostChildren, arcs, labeled):
		"""This function should return a Transition object representing the correct action to
		to take according to the oracle."""
	#	print stack, buff, leftmostChildren, rightmostChildren, arcs, labeled
		if len(stack) <= 1:
			return Transition(Transition.Shift, None)
	#	print stack,
	#	print stack[-2][0] in leftmostChildren[stack[-1][0]].keys(), self.all_children_are_attached(stack[-2], buff, rightmostChildren)
		if stack[-2][0] in leftmostChildren[stack[-1][0]].keys() and self.all_children_are_attached(stack[-2], buff, rightmostChildren):
	#		print 'left' 
			return Transition(Transition.LeftArc, leftmostChildren[stack[-1][0]][stack[-2][0]] if labeled else None)
		if stack[-1][0] in rightmostChildren[stack[-2][0]].keys() and self.all_children_are_attached(stack[-1], buff, rightmostChildren):
	#		print 'right'
			return Transition(Transition.RightArc, rightmostChildren[stack[-2][0]][stack[-1][0]] if labeled else None)
	#	print "shift"
		return Transition(Transition.Shift, None)
		#assert False, 'Please implement this function!'

	def all_children_are_attached(self, word, buff, rightmostChildren):
	#	print len(buff) == 0, len(rightmostChildren[word[0]]) == 0, buff[-1][0], rightmostChildren[word[0]],
		return len(buff) == 0 or len(rightmostChildren[word[0]]) == 0 or int(buff[-1][0]) == 0 or int(buff[-1][0]) > max([int(child) for child in rightmostChildren[word[0]].keys()])