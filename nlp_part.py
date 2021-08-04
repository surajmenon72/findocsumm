import numpy as np
import string
import copy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding_size = 384

s_test = ['Research and development', 'Selling, general, and administrative', 'Sales and marketing']

revenue_tree = [[['Total Revenues', 'Total Sales']], []]
cogs_tree = [[['Cost of goods sold', 'Cost of sales']], []]
opex_tree = [[['Operating expenses', 'Operating costs']], [['Research and development'], ['Selling, general, and administrative'], ['Sales and marketing']]]

trees = [revenue_tree, cogs_tree, opex_tree]

def bucket_headers(headers, header_labels):
	#do some pre-embeddings of the headers, and then search w/ matrix multiplication later
	num_headers = len(headers)
	header_embeddings = np.zeros((num_headers, embedding_size))
	count_embeddings =  np.zeros((num_headers, embedding_size))

	for i, header in enumerate(headers):
		((start_X, start_Y, end_X, end_Y), text) = header
		e = model.encode(text)
		if (header_labels[i, 1] == 0):
			header_embeddings[i, :] = copy.deepcopy(e)
		else:
			count_embeddings[i, :] = copy.deepcopy(e)

	matches = []
	distance_threshold = 10 #TODO: Tune
	for i, t in enumerate(trees): #tree
		tree_match = []
		for j, l in enumerate(t): #layer
			list_match = []
			if (len(l) > 0):
				for k, p in enumerate(l): #synonym
					p_embed = model.encode(p)
					num_syn = len(p)
					header_distances = np.zeros((num_headers, num_syn))
					count_distances = np.zeros((num_headers, num_syn))

					header_distances = np.sum(np.dot(header_embeddings, p_embed.T), axis=1)
					s_header_distances = np.divide(header_distances, header_labels[:, 0])
					count_distances = np.sum(np.dot(count_embeddings, p_embed.T), axis=1)
					
					max_index = 1e6
					if (j == 0): #favor header
						if (np.amax(s_header_distances) >= distance_threshold):
							max_index = np.argmax(s_header_distances)
						elif (np.amax(count_distances) >= distance_threshold):
							max_index = np.argmax(count_distances)
					else: #favor count
						if (np.amax(count_distances) >= distance_threshold):
							max_index = np.argmax(count_distances)
						elif (np.amax(s_header_distances) >= distance_threshold):
							max_index = np.argmax(s_header_distances)

					if (max_index != 1e6):
						header_append = headers[max_index]
						((start_X, start_Y, end_X, end_Y), text) = header_append
						print (text)
						list_match.append(text)
					else:
						list_match.append('')
			tree_match.append(list_match)
		matches.append(tree_match)

	print ('Came up with these matches')
	print (matches)


# def embedding_sentences(sentences):
# 	s_embeddings = model.encode(sentences)

# 	for sentence, embedding in zip(sentences, s_embeddings):
# 		print("Sentence:", sentence)
# 		print("Embedding:", embedding)
# 		print("")

# 	return s_embeddings

# print ('Starting')
# e = embedding_sentences(s_test)

# e1 = np.dot(e[0], e[1])
# e2 = np.dot(e[0], e[2])
# e3 = np.dot(e[1], e[2])

# print (e1)
# print (e2)
# print (e3)


