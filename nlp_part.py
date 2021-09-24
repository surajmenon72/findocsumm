import numpy as np
import pandas as pd #need openpyxl as well
import string
import copy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding_size = 384

revenue_tree = [[['Revenues', 'Sales']], []]
cogs_tree = [[['Cost of goods sold', 'Cost of sales']], []]
opex_tree = [[['Operating expenses', 'Operating costs']], [['Research and development'], ['Selling, general, and administrative'], ['Sales and marketing']]]
income_tree = [[['Operating income', 'Operating profit']], []]
non_operating_tree = [[['Non-Operating income', 'Non-Operating income']], []]

trees = [revenue_tree, cogs_tree, opex_tree, income_tree, non_operating_tree]

def label_headers(headers, clean_results):
	num_headers = len(headers)
	header_labels = np.zeros((num_headers, 2))
	num_cols = clean_results.shape[1]-1
	col_thresh = num_cols-1 #TODO: Verify, but allow one zero
	zero_rows = []
	c = 1
	for header in headers:
		((start_X, start_Y, end_X, end_Y), text) = header
		arr = clean_results[c, :]
		count = (np.count_nonzero(arr)-1) #include the column header, except for first row
		if (count < col_thresh):
			zero_rows.append(c-1)
		c += 1

	indent_thresh_zero = 5 #TODO: tune
	indent_thresh_norm = 25
	header_level = 1
	current_indent = 0
	prev_row_zero = False
	for i, header in enumerate(headers):
		((start_X, start_Y, end_X, end_Y), text) = header
		if (i == 0):
			header_labels[i, 0] = header_level
			current_indent = start_X
		else:
			thresh = 0
			if (current_indent == 0):
				thresh = indent_thresh_zero
			else:
				thresh = indent_thresh_norm

			if ((start_X - current_indent) >= thresh):
				header_level += 1
			elif ((start_X - current_indent) <= -thresh):
				header_level = 1
			else: #no indent
				if (i in zero_rows):
					header_level = 1
				elif (prev_row_zero == True):
					if (i not in zero_rows):
						header_level += 1
		
			header_labels[i, 0] = header_level
			current_indent = start_X

		if (i not in zero_rows):
			header_labels[i, 1] = 1
			prev_row_zero = False
		else:
			prev_row_zero = True

	print ('Printing Headers')
	for i, header in enumerate(headers):
		print (header)
		print (header_labels[i, 0])

	return header_labels

def print_tree(matches):
	for i, t in enumerate(matches):
		ct = trees[i]
		for j, l in enumerate(t):
			cl = ct[j]
			for k, p in enumerate(l):
				cp = cl[k]
				
				print ('Tree Part')
				print (cp)
				print ('Match')
				print (p)

def print_final_tree(matches):
	for i, t in enumerate(matches):
		ct = trees[i]
		for j, l in enumerate(t):
			cl = ct[j]
			for k, p in enumerate(l):
				cp = cl[k]
				match, suggest, sub_match = p
				
				print ('Tree Part')
				print (cp)
				print ('Match')
				print (match)
				print ('Suggest')
				print (suggest)
				print ('Sub-Match')
				print (sub_match)

def match_top(headers, header_labels):
	num_headers = len(headers)
	header_embeddings = np.zeros((num_headers, embedding_size))

	for i, header in enumerate(headers):
		((start_X, start_Y, end_X, end_Y), text) = header
		e = model.encode(text)
		header_embeddings[i, :] = copy.deepcopy(e)

	matches = []
	distance_difference_threshold = 15 #TODO: Tune this
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
						if (header_labels[h, 0] == (j+1)): #match the layer, j+1
							level_best.append((h, header_distances[h]))
						else:
							nonlevel_best.append((h, header_distances[h]))
					c = 0
					for h, dist in level_best:
						nl_h = nonlevel_best[c][0]
						nl_dist = nonlevel_best[c][1]
						if ((nl_dist - dist) < distance_difference_threshold):
							header_to_append = headers[h]
							((start_X, start_Y, end_X, end_Y), text) = header_to_append
							list_match.append((h, dist, text))
						else:
							header_to_append = headers[nl_h]
							((start_X, start_Y, end_X, end_Y), text) = header_to_append
							list_match.append((nl_h, dist, text))
							c += 1

						if (len(list_match) >= num_matches):
							break
					layer_match.append(list_match)
			tree_match.append(layer_match)
		matches.append(tree_match)

	# print ('')
	# print ('Match Top!')
	# print ('')
	# print_tree(matches)

	return matches

def compute_confidence(pot_headers):
	#TODO: make this more efficient...
	all_distances = []
	for i, t in enumerate(pot_headers):
		for j, l in enumerate(t):
			for k, p in enumerate(l):
				for h, dist, text in p:
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
			for k, p in enumerate(l):
				list_match = []
				for h, dist, text in p:
					conf = n_all_distances[c]
					list_match.append((h, dist, conf, text))
					c += 1
				layer_match.append(list_match)
			tree_match.append(layer_match)
		new_pot_headers.append(tree_match)

	# print ('')
	# print ('Set Confidence!')
	# print ('')
	# print_tree(new_pot_headers)

	return new_pot_headers

def is_marked(phrases, tree):
	marked = False
	marked_threshold = 30
	for s in phrases:
		s_e = model.encode(s)
		for m in tree:
			for m_l in m:
				lm, ls, lsub = m_l
				for a in lsub:
					a_h, a_conf, a_text = a
					#a_text = a_text.rstrip('\n\x0c')
					a_e = model.encode(a_text)
					if (np.dot(s_e, a_e) >= marked_threshold):
						#print ('MARKED')
						#print (s)
						#print (a_text)
						marked = True
	return marked

def match_bottom(pot_headers, headers, header_labels):
	num_headers = len(headers)
	final_selections = []
	conf_thresh = .40 #TODO: Tune
	for i, t in enumerate(pot_headers):
		ct = trees[i]
		tree_match = []
		for j, l in enumerate(t):
			cl = ct[j]
			layer_match = []
			for k, p in enumerate(l):
				cp = cl[k]
				marked = is_marked(cp, tree_match)

				if (marked == True): #TODO: for now skipping, but we could change to suggestions
					layer_match.append(([], [], []))
					continue

				list_match = []
				list_suggest = []
				sub_list_match = []
				for h, dist, conf, text in p:
					matched = False
					if (conf > conf_thresh):
						if (len(list_match) == 0):
							matched = True
							list_match.append((h, conf, text))
						# else:
						# 	list_suggest.append((h, conf, text))
					else:
						list_suggest.append((h, conf, text))

					if (matched == True): #put in sublevels
						done = False
						start_level = header_labels[h, 0]
						index_it = h
						while (done == False):
							if ((index_it+1) < num_headers):
								index_it += 1
								new_level = header_labels[index_it, 0]
								if ((new_level <= start_level) or (new_level == 1)):
									done = True
								else:
									header_to_append = headers[index_it]
									((start_X, start_Y, end_X, end_Y), new_text) = header_to_append
									sub_list_match.append((index_it, conf, new_text))
							else:
								done = True
				layer_match.append((list_match, list_suggest, sub_list_match))
			tree_match.append(layer_match)
		final_selections.append(tree_match)

	# print ('')
	# print ('Set Final!')
	# print ('')
	# print_final_tree(final_selections)

	return final_selections


def fit_hierarchy(pot_headers, headers, header_labels):
	num_headers = len(headers)
	cleaned_selection = []
	for i, t in enumerate(pot_headers):
		tree_match = []
		for j, l in enumerate(t):
			layer_match = []
			top_level_matches = [] #assume for now the top level has only one concept
			possible_sub_matches = []
			for k, p in enumerate(l):
				match, suggest, sub_match = p
				if (j == 0): #set the header label, assuem just one
					if (len(match) > 0):
						h, conf, text = match[0]
						if (header_labels[h, 0] == 1):
							top_level_matches.append(h)
							done = False
							index_it = h+1
							while (done == False):
								if (index_it < num_headers):
									if (header_labels[index_it, 0] > 1):
										possible_sub_matches.append(index_it)
									else:
										done = True
								else:
									done = True
								index_it += 1
				else: #prune based on header label
					#check match, suggests
					if (len(match) > 0):
						h, conf, text = match[0]
						if (h not in possible_sub_matches):
							suggest.insert(0, (match[0])) #for now just clear matches and add to suggest
							match = []
					if (len(suggest) > 0):
						h, conf, text = suggest[0]
						if (h not in possible_sub_matches):
							#suggest = []
							match = []

				layer_match.append((match, suggest, sub_match))
			tree_match.append(layer_match)
		cleaned_selection.append(tree_match)

	return cleaned_selection


def bucket_headers(headers, header_labels):

	pot_headers = match_top(headers, header_labels)

	pot_headers = compute_confidence(pot_headers)

	final_selections = match_bottom(pot_headers, headers, header_labels)

	final_selections = fit_hierarchy(final_selections, headers, header_labels)

	return final_selections

def print_buckets(bucketed_headers, header_labels, headers, dates, dates_full, counts, date_contexts, count_contexts, clean_results, filename):
	headers_text = []

	#TODO: make this less stupid
	start_header = ''
	if (len(count_contexts) > 0):
		cc = count_contexts[0]
		((start_X, start_Y, end_X, end_Y), text) = cc
		start_header = text

	headers_text.append(start_header)

	header_order = []
	for i, t in enumerate(bucketed_headers):
		ct = trees[i]
		for j, l in enumerate(t):
			cl = ct[j]
			for k, p in enumerate(l):
				cp = cl[k][0]
				match, suggest, sub_match = p
				if (len(match) > 0): #list match
					h, conf, t = match[0]
					pot_header = headers[h]
					((start_X, start_Y, end_X, end_Y), text) = pot_header
					header_text = cp + ' --MATCH-- ' + text
					header_order.append(h)
					headers_text.append(header_text)

					for s, s_conf, s_t in sub_match:
						pot_header = headers[s]
						((start_X, start_Y, end_X, end_Y), text) = pot_header
						header_text = 'SUBLEVEL-SUGGEST-- ' + text 
						header_order.append(s)
						headers_text.append(header_text)
				elif (len(suggest) > 0):
					h, conf, t = suggest[0]
					pot_header = headers[h]
					((start_X, start_Y, end_X, end_Y), text) = pot_header
					header_text = cp + ' --SUGGEST-- ' + text
					header_order.append(h)
					headers_text.append(header_text)
					
					for s, s_conf, s_t in sub_match:
						pot_header = headers[s]
						((start_X, start_Y, end_X, end_Y), text) = pot_header
						header_text = 'SUBLEVEL-SUGGEST-- ' + text 
						header_order.append(s)
						headers_text.append(header_text)


	df = pd.DataFrame({'Headers': headers_text})

	date_cols = clean_results[0, :]
	num_cols = clean_results.shape[1]

	for c in range(1, num_cols):
		ind_dc = int(date_cols[c])
		date = dates[ind_dc]
		date_full = dates_full[ind_dc]
		c_text = ''
		if (len(date_contexts) > 0):
			ctxt = date_contexts[c-1]
			if (ctxt):
				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = ctxt
				c_text = textc
		date_str = str(c) + '-' + str(date_full[0]) + '/' + str(date_full[1]) + '/' + str(date_full[2]) + '--' + c_text

		#fill in values
		col_arr = clean_results[1:, c]
		values = []
		values.append('--') #dummy for first row
		for i in header_order:
			if (header_labels[i, 1] == 1): #check if a zero row
				val_ind = int(col_arr[i])
				count = counts[val_ind]
				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count
				values.append(float(textc))
			else:
				values.append('--')

		#add to dateframe
		df[date_str] = values

	#print (df)
	#df.to_excel(filename, sheet_name='sheet1', index=False)
	df.to_csv(filename, index=False)
