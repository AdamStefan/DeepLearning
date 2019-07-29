import options
from word2vec import word2vec
settings = options.settings

text = "natural language processing and machine learning is fun and exciting"

# Note the .lower() as upper and lowercase does not matter in our implementation
# [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
corpus = [[word.lower() for word in text.split()]]

print(corpus)



# Initialise object
w2v = word2vec(settings)
# Numpy ndarray with one-hot representation for [target_word, context_words]
w2v.load_data(corpus)  

w2v.train()