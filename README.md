## SentenceTransformers Documentation (excerpt) 
https://sbert.net/

Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models (quickstart). This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

## A few areas of interest

* Can we cluster wine reviews based on S-BERT embeddings?

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/umap cluster viz.png" width=50% height=50%>

Despite rather adequate within cluster coherence, the Silhouette score is quite low, indicating weak clustering. 
```
Silhouette Score: 0.024117592722177505
Weak clustering
```
Additionally, the Top Words per cluster do not seem to be much different, indicating that most reviews use a standard bag of descriptors.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/top_words.png" width=20% height=20%>

Let's now take a look at other considerations with this dataset.

* Can we leverage S-BERT embeddings to create a recommender system for "customers" that have tried several wines?

## S-Bert Embeddings Wine Recommender

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

Using Averaged Embeddings (Description, Variety, and Title), for the preferred wine with title 'La Castellina 2007 Squarcialupi Riserva  (Chianti Classico)' and Variety: 'Sangiovese', the recommendations are shown below. However, the final results still aren't great recommendations for the nuanced wine lover.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/reccs.png" width=85% height=85%>

## RAG (Retrieval-Augmented Generation) with Langchain, OpenAI, and FAISS

* Langchain is an open source Python framework used to simplify the creations of applications, more specifically to integrate LLM API's and user prompts (questions about underlying documents).
* FAISS stands for Facebook AI Similarity Search and can also be used to create vector embeddings for documents and to perform similarity search operations.
* OpenAI is the large language model that is used for querying the document vectors

This allows for more nuanced document retrieval system rather than just matching specific wines based on a description. Let's take a look at a few examples:

If we ask the model to "Suggest some Italian wines that are earthy" and then we use a similar weighting methodology (for Price and Rating), the results look fairly appropriate.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/earthy_italian.png" width=85% height=85%>

Now let's say we want to ask the model something more ambiguous like "What if I want to drink Rose all day?", Below are the results with k=3 specified.

<img src="https://github.com/mwheeler235/wine-reviews/blob/main/img/rose_all_day.png" width=85% height=85%>

These certainly do appear to be great options for "Rose ALL DAY" ;) Of course, Vintage Year would be a very important consideration if we had access to more recent reviews. Overall, this RAG approach is more flexible and more accurate for retrieving wines based on nuanced preferences, i.e. for humans being human!
