# SentenceTransformers Documentation (excerpt) 
https://sbert.net/

Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models (quickstart). This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

# A few areas of interest

* Can we cluster wine reviews based on S-BERT embeddings?

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/umap cluster viz.png" width=50% height=50%>

Despite rather adequate cluster coherence, the Top Words per cluster do not seem to be much different, indicating that most reviews use a standard bang of descriptors.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/top_words.png" width=20% height=20%>

Let's now take a look at other considerations with this dataset.

* Can we leverage S-BERT embeddings to create a recommender system for "customers" that have tried several wines?

# Wine Recommender

Relying solely on Description embeddings tends to recommend wines from tasters with similar lexicons. As such, we can create embeddings for Variety and also for Title, then combine these with a weighted average. In the average, we can tweak the preference to favor Variety embeddings since the variety of a wine more closely aligns with a customer's flavor palate (rather than the title or the description). 
```
combined_embeddings = ((1/2)*corpus_embeddings + 2*corpus_variety_embeddings + corpus_title_embeddings) / 3
```

Then, finally, we can tweak the recommendation to weight a lower price and a higher rating.
```
weight_points = .5
weight_price = .5
df_user0_reccs2['weighted_customer_value'] = (df_user0_reccs2['points'] * weight_points) + -1*(df_user0_reccs2['price'] * weight_price)
```

Using Averaged Embeddings (Description, Variety, and Title), for the preferred wine with title 'La Castellina 2007 Squarcialupi Riserva  (Chianti Classico)' and Variety: 'Sangiovese', the recommendations are shown below.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/reccs.png" width=85% height=85%>
