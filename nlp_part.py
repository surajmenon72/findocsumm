import numpy as np
import string
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

headers = ['Total Revenues', 'Total Costs', 'Elegy of Emptiness', 'Total Sales']

def embedding_sentences(sentences):
	s_embeddings = model.encode(sentences)

	for sentence, embedding in zip(sentences, s_embeddings):
		print("Sentence:", sentence)
		print("Embedding:", embedding)
		print("")

	return s_embeddings

print ('Starting')
e = embedding_sentences(headers)

e1 = np.dot(e[0], e[1])
e2 = np.dot(e[0], e[2])
e3 = np.dot(e[1], e[2])
e4 = np.dot(e[0], e[3])

print (e1)
print (e2)
print (e3)
print (e4)

