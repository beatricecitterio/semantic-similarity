# **Assignment 2 - Semantic Similarity**
## **Track 1** - Discrete Representation
In the first track I used TfidfVectorizer with the following parameters: <code>{max_features=20000,    ngram_range=(1, 3), sublinear_tf=True, analyzer='char_wb', min_df=1, max_df=0.95}</code>.


These parameters were found by running a thorough hyperparameter search. To assess performance, the vectorizer is trained on <code>train_prompts.csv</code> and evaluated on <code>dev_prompts.csv</code>, reaching an average BLEU score of 0.0885. 

Preprocessing was not useful, therefore is not included. The only transformation I applied to the train data frame was to exclude invalid responses in the matching phase.

Note that the most similar prompt is found using Nearest Neighbors with cosine similarity and 1 neighbor. 

## **Track 2** - Continuous Representation
In the second track I used FastText with the following parameters: <code>{'vector_size': 100, 'window': 10, 'min_count' = 1, 'epochs': 100, 'sg': 1}</code>, yielding average BLEU score of 0.0866. Again, I chose the parameters based on the optimal BLEU score found by running an extensive hyperparameter search. 

In this track as well I excluded invalid responses in the matching phase.

Again, the most similar prompt is found using Nearest Neighbors with cosine similarity and 1 neighbor. 

## **Track 3** - Open Text Representation
In the final track, I used the all-mpnet-base-v2 SentenceTransformer model, as it had the best performance. Other models I tried include all-MiniLM-L6-v2, all-MiniLM-L12-v2, paraphrase-MiniLM-L6-v2, paraphrase-mpnet-base-v2. 

As before, I loaded the datasets, removed invalid responses from train, and encoded all prompts into dense vector embeddings, which were then normalized. To retrieve most similar prompt, I used again Nearest Neighbors with cosine similarity and one neighbor. The setup achieved a BLEU score of 0.1080. 