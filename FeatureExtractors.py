from collections import defaultdict
import sys

class BasicJointFeatureExtractor:
	def extract_features(self, transition, stack, buff, arcs, labels, previous_transitions):
		features = defaultdict(float)

		tType = transition.transitionType
		label = transition.label

		# Top two POS tags from the stack
		for i in range(2):
			if i >= len(stack):
				break
			s = stack[-(i+1)]
			pos = s[3]
			features['transition=%d,s%d.pos=%s' % (tType, i, pos)] = 1
			if self.labeled:
				features['label=%s,s%d.pos=%s' % (label, i, pos)] = 1

		# Next four POS tags from the buffer
		for i in range(4):
			if i >= len(buff):
				break
			b = buff[-(i+1)]
			pos = b[3]
			features['transition=%d,b%d.pos=%s' % (tType, i, pos)] = 1
			#if self.labeled:
			#	features['label=%s,b%d.pos=%s' % (label, i, pos)] = 1

		# Previous transition type
		if False and len(previous_transitions) > 0:
			prev = previous_transitions[-1].transitionType
			print prev
			features['transition=%d,prev_transition=%d' % (tType, prev)] = 1
			if self.labeled:
				features['label=%s,prev_transition=%d' % (label, prev)] = 1
		else:
			features['transition=%d,prev_transition=None' % (tType)] = 1
			if self.labeled:
				features['label=%s,prev_transition=None' % (label)] = 1

		# Bias feature
		features['transition=%d' % (tType)] = 1

		if self.labeled:
			# Action and label pair
			features['transition=%d,label=%s' % (tType, label)] = 1
			# Label bias
			features['label=%s' % (label)] = 1

		return features
		
class BasicFeatureExtractor:
	def extract_features(self, transition, stack, buff, arcs, labels, previous_transitions):
		features = defaultdict(float)

		# Top four words and POS tags from the stack
		for i in range(4):
			if i >= len(stack):
				break
			s = stack[-(i+1)]
			pos = s[3]
			features['s%d.pos' % (i)] = str(pos)
			word = s[1].lower()
			features['s%d.word' % (i)] = str(word)

		# Next four words POS tags from the buffer
		for i in range(4):
			if i >= len(buff):
				break
			b = buff[-(i+1)]
			pos = b[3]
			features['b%d.pos' % (i)] = str(pos)
			word = b[1].lower()
			features['b%d.word' % (i)] = str(word)
				
		# Previous transition type
		if len(previous_transitions) > 0:
			prevT = previous_transitions[-1].transitionType
			prevL = previous_transitions[-1].label
			features['prev_transition'] = str(prevT)
			features['prev_transition_label'] = str(prevL)
			

		# Number of left and right children the top two things on the stack already have
		for i in range(2):
			if i >= len(stack):
				break
			s = stack[-(i+1)]
			left = [k for k in arcs if arcs[k] == s[0] and int(k) < int(s[0])]
			right = [k for k in arcs if arcs[k] == s[0] and int(s[0]) < int(k)]
			features['s%d.left_len' % (i)] = len(left)
			features['s%d.right_len' % (i)] = len(right)
			#~ for j in range(len(left)):
				#~ if pos in self.pos_vocab:
					#~ features['s%d.pos' % (i)] = self.pos_vocab[pos]
				#~ else:
					#~ self.pos_vocab[pos] = len(self.pos_vocab)
					#~ features['s%d.pos' % (i)] = self.pos_vocab[pos]

		#~ print >>sys.stderr, features
		return features