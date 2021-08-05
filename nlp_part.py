import numpy as np
import string
import copy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding_size = 384

revenue_tree = [[['Total Revenues', 'Total Sales']], []]
cogs_tree = [[['Cost of goods sold', 'Cost of sales']], []]
opex_tree = [[['Operating expenses', 'Operating costs']], [['Research and development'], ['Selling, general, and administrative'], ['Sales and marketing']]]
income_tree = [[['Operating Income', 'Total Income']], []]

trees = [revenue_tree, cogs_tree, opex_tree, income_tree]

def match_top(headers, header_labels):
	#do some pre-embeddings of the headers, and then search w/ matrix multiplication later
	num_headers = len(headers)
	header_embeddings = np.zeros((num_headers, embedding_size))
	#count_embeddings =  np.zeros((num_headers, embedding_size))

	for i, header in enumerate(headers):
		((start_X, start_Y, end_X, end_Y), text) = header
		e = model.encode(text)
		header_embeddings[i, :] = copy.deepcopy(e)

	matches = []
	distance_difference_threshold = 50 #TODO: Tune this
	num_matches = 3
	for i, t in enumerate(trees): #tree
		tree_match = []
		for j, l in enumerate(t): #layer
			layer_match = []
			if (len(l) > 0):
				for k, p in enumerate(l): #synonym
					list_match = []
					p_embed = model.encode(p)
					num_syn = len(p)
					header_distances = np.zeros((num_headers, num_syn))

					header_distances = np.sum(np.dot(header_embeddings, p_embed.T), axis=1)/num_syn
					s_header_distances = np.flip(np.argsort(header_distances, axis=0), axis=0)

					#Manner of taking into account different levels
					level_best = []
					nonlevel_best = []
					for h in s_header_distances:
						if (header_labels[h, 0] == j):
							level_best.append((h, header_distances[h]))
						else:
							nonlevel_best.append((h, header_distances[h]))
					c = 0
					for h, dist in level_best:
						nl_h = nonlevel_best[c][0]
						nl_dist = nonlevel_best[c][1]
						if ((dist - nl_dist) > distance_difference_threshold):
							list_match.append((h, dist))
						else:
							list_match.append((nl_h, dist))
							c += 1

						if (len(list_match) > num_matches):
							break
				layer_match.append(list_match)
			tree_match.append(layer_match)
		matches.append(tree_match)
	print ('Came up with these matches')
	print (matches)
	return matches

def compute_confidence(pot_headers):
	#TODO: make this more efficient...
	all_distances = []
	for i, t in enumerate(pot_headers):
		for j, l in enumerate(t):
			for k, p in enumerate(l):
				for h, dist in p:
					all_distances.append(dist)

	a_all_distances = np.array(all_distances)
	max_dist = np.amax(a_all_distances)
	n_all_distances = a_all_distances/max_dist #TODO: consider additional factors

	new_pot_headers = []
	c = 0
	for i, t in enumerate(pot_headers):
		tree_match = []
		for j, l in enumerate(t):
			layer_match = []
			for k, p in enumerate(l)
				list_match = []
				for h, dist in p:
					conf = n_all_distances[c]
					list_match.append((h, dist, conf))
					c += 1
				layer_match.append(list_match)
			tree_match.append(layer_match)
		new_pot_headers.append(tree_match)

	return new_pot_headers

def match_bottom(pot_headers, headers, header_labels):

	num_headers = len(headers)
	final_selections = []
	conf_thresh = .50 #TODO: Tune
	for i, t in enumerate(pot_headers):
		tree_match = []
		for j, l in enumerate(t):
			layer_match = []
			for k, p in enumerate(l):
				list_match = []
				list_suggest = []
				sub_list_match = []
				for h, dist, conf in p:
					matched = False
					if (conf > conf_thresh):
						if (len(list_match) == 0):
							matched = True
							list_match.append((h, conf))
						else:
							list_suggest.append((h, conf))
					else:
						list_suggest.append((h, conf))

					if (matched == True): #put in sublevels
						done = False
						start_level = header_labels[h, 0]
						index_it = h
						while (done == False):
							if ((index_it+1) < num_headers):
								index_it += 1
								new_level = header_labels[index_it, 0]
								if ((new_level < start_level) or (new_level == 1)):
									done = True
								else:
									sub_list_match.append((h, conf))
							else:
								done = True
				layer_match.append((list_match, list_suggest, sub_list_match))
			tree_match.append(layer_match)
		final_selections.append(tree_match)

	return final_selections

def bucket_headers(headers, header_labels):

	pot_headers = match_top(headers, header_labels)

	pot_headers = compute_confidence(pot_headers)

	final_selections = match_bottom(pot_headers, headers, header_labels)

	return final_selections




