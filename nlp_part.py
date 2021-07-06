import numpy as np
import sister
import string

buckets_counts = ['Total Revenues', 'Total Costs', 'Elegy of Emptiness']

def bucket_results(headers, dates, dates_full, counts, date_contexts, count_contexts, clean_results):
	print ('Bucketing Results')
	embedder = sister.MeanEmbedding(lang="en")
	num_buckets = len(buckets_counts)
	embedding_size = 300 #given by sister

	bucket_embeddings = np.zeros((num_buckets, embedding_size))
	i = 0
	for bucket in buckets_counts:
		bucket_embeddings[i, :] = embedder(bucket) 
		i += 1

	num_headers = len(headers)
	header_embeddings = np.zeros((num_headers, embedding_size))
	i = 0
	for header in headers:
		((start_X, start_Y, end_X, end_Y), text) = header
		text_c = text.translate(str.maketrans({key: " ".format(key) for key in string.punctuation}))
		header_embeddings[i, :] = embedder(text_c)
		i += 1

	embedding_calc = np.matmul(header_embeddings, bucket_embeddings.T)
	embedding_max = np.argmax(embedding_calc, axis=1)

	#print bucket results
	for i in range(len(headers)):
		print ('Match')
		print (headers[i])
		ind = embedding_max[i]
		print (buckets_counts[ind])







