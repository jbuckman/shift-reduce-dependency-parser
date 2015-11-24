import sys
import math
from Transition import Transition
from collections import defaultdict
from copy import copy
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from FeatureExtractors import BasicFeatureExtractor, BasicJointFeatureExtractor

def dot(features, weights):
	score = 0.0
	for key in set(features) & set(weights):
		score += features[key] * weights[key]
	return score

class Model:
	"""This is the base class for a model, that other models inherit from."""
	def __init__(self, labeled):
		self.labeled = labeled
		self.learning_rate = 1.0
		self.weights = defaultdict(float)
		self.label_set = Transition.ALL_LABELS if labeled else [None]
		self.initialize()

	def initialize(self):
		pass

	def possible_transitions(self, stack, buff):
		possible_transitions = []
		if len(buff) >= 1:
			possible_transitions.append(Transition(Transition.Shift, None))
		if len(stack) >= 2:
			for label in self.label_set:
				possible_transitions.append(Transition(Transition.LeftArc, label))
				possible_transitions.append(Transition(Transition.RightArc, label))
		#assert len(possible_transitions) > 0
		return possible_transitions

	def all_possible_transitions(self):
		possible_transitions = []
		possible_transitions.append(Transition(Transition.Shift, None))
		for label in self.label_set:
			possible_transitions.append(Transition(Transition.LeftArc, label))
			possible_transitions.append(Transition(Transition.RightArc, label))
		#assert len(possible_transitions) > 0
		return possible_transitions


class OnlineModel(Model):
	"""General stuff for training online models"""
	def learn(self, correct_transition, stack, buff, arcs, labels, previous_transitions):
		correct_features = None
		best_features = None
		best_score = None
		best_transition = None
		for transition in self.possible_transitions(stack, buff):
			features = self.extract_features(transition, stack, buff, arcs, labels, previous_transitions)
			score = dot(features, self.weights)
			if best_score == None or score > best_score:
				best_score = score
				best_transition = transition
				best_features = features
			if transition == correct_transition:
				correct_features = features

		if best_transition != correct_transition:
			assert best_features != None
			assert correct_features != None
			self.update(correct_features, best_features)
	
	def build_model(self):
		pass
		
	def predict(self, stack, buff, arcs, labels, previous_transitions):
		best_score = None
		best_transition = None
		for transition in self.possible_transitions(stack, buff):
			features = self.extract_features(transition, stack, buff, arcs, labels, previous_transitions)
			score = self.score(features)
			if best_score == None or score > best_score:
				best_score = score
				best_transition = transition
		return (best_score, best_transition)
		
	def predict_all(self, stack, buff, arcs, labels, previous_transitions):
		best_choices = []
		i = 9999999999999999999
		orderer = {}
		for transition in self.possible_transitions(stack, buff):
			i -= 1
			orderer[transition] = i
			features = self.extract_features(transition, stack, buff, arcs, labels, previous_transitions)
			score = self.score(features)
			transition.score = score
			best_choices.append((score, transition))
		return sorted(best_choices, reverse=True, key=lambda x:(x[0], orderer[x[1]]))


class BatchModel(Model):
	def initialize(self):
		self.X_features = []
		self.x_vectorizer = DictVectorizer()
		self.Y = []
		
	def learn(self, correct_transition, stack, buff, arcs, labels, previous_transitions):
		features = self.extract_features(correct_transition, stack, buff, arcs, labels, previous_transitions)
		self.X_features.append(features)
		self.Y.append(correct_transition.to_category())
	
	def build_model(self):
		X = self.x_vectorizer.fit_transform(self.X_features)
		Y = self.Y
		self.internal_model = self.create_model(X, Y)
		print >>sys.stderr, "model built"
		
	def predict(self, stack, buff, arcs, labels, previous_transitions):
		features = self.extract_features(None, stack, buff, arcs, labels, previous_transitions)
		X_i = self.x_vectorizer.transform(features)
		prediction = self.internal_model.predict(X_i)
		return Transition.from_category(prediction)
		
	def predict_all(self, stack, buff, arcs, labels, previous_transitions):
		features = self.extract_features(None, stack, buff, arcs, labels, previous_transitions)
		X_i = self.x_vectorizer.transform(features)
		scores = self.scores(X_i, self.possible_transitions(stack, buff))
		#~ print >>sys.stderr, " choosing %s\n" % (str(sorted(scores, reverse=True)))
		return sorted(scores, reverse=True)


class PerceptronModel(OnlineModel, BasicJointFeatureExtractor):
	"""This is a simple perceptron."""
	def update(self, correct_features, predicted_features):
		keys = set(correct_features) | set(predicted_features)
		for key in keys:
			c = correct_features.get(key, 0.0)
			p = predicted_features.get(key, 0.0)
			self.weights[key] += (c - p) * self.learning_rate
			if self.weights[key] == 0.0:
				del self.weights[key]
				
	def score(self, features):
		return dot(features, self.weights)
		
class SVCModel(BatchModel, BasicFeatureExtractor):
	def create_model(self, X, Y):
		model = svm.SVC(decision_function_shape='ovr')
		model.fit(X, Y)
		return model
	
	def scores(self, X_i, possible_transitions=None):
		probs = self.internal_model.decision_function(X_i)[0]
		ans = []
		for i in range(len(self.internal_model.classes_)):
			transition = Transition.from_category(self.internal_model.classes_[i], score=probs[i])
			if possible_transitions == None or transition in possible_transitions:
				ans.append((probs[i], transition))
		return ans
	
class RandomForestModel(BatchModel, BasicFeatureExtractor):
	def create_model(self, X, Y):
		model = RandomForestClassifier(n_estimators=10)
		model.fit(X, Y)
		return model
		
	def scores(self, X_i, possible_transitions=None):
		probs = self.internal_model.predict_log_proba(X_i)[0]
		ans = []
		for i in range(len(self.internal_model.classes_)):
			transition = Transition.from_category(self.internal_model.classes_[i], score=probs[i])
			if possible_transitions == None or transition in possible_transitions:
				ans.append((probs[i], transition))
		return ans
					
class SKPerceptronModel(BatchModel, BasicFeatureExtractor):
	def create_model(self, X, Y):
		model = linear_model.Perceptron()
		model.fit(X, Y)
		return model
		
	def scores(self, X_i, possible_transitions=None):
		probs = self.internal_model.decision_function(X_i)[0]
		ans = []
		for i in range(len(self.internal_model.classes_)):
			transition = Transition.from_category(self.internal_model.classes_[i], score=probs[i])
			if possible_transitions == None or transition in possible_transitions:
				ans.append((probs[i], transition))
		return ans