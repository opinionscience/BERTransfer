# BERTransfer
<img src="https://raw.githubusercontent.com/Pclanglais/BERTransfer/main/BERTransfer.png" style="float:right;" alt="Bertransfer logo"  width="200"/>

BERTransfer is a text mining application that make it possible to apply topics defined for one corpus to another corpus.

BERTransfer is built on top of [BERTopic](https://github.com/MaartenGr/BERTopic) and is part of the same ecosystem of BERT-based tools for text classification. BERTopic is a topic modeling technique that leverages 🤗 transformers and a custom class-based TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. BERTransfer makes it possible to reuse topics defined by BERTopic on one corpus to further corpora. 

BERTransfer help create annotated classification models from BERTopic using the following workflow:
* Topics are defined from an initial corpus.
* Topics can be annotated based on several features: the characteristic words and a list of representative documents.
* Topic annotations can be transfered to a new corpus.

## Use cases

BERTransfer has been used effectively in following tasks:
* Continuous observation of new data, such as an online conversation
* Dealing with very large corpus. In its default setting, BERTopic takes a long time on corpus larger than 70,000 documents. Topic transfer is a possible solution: topics have been successfully applied for corpus of millions of documents.
* Reuse complex annotations that may require the interpretation of an expert analyst.

## Creation of topic models

This process is very close to the current workflow of BERTopic. BERTransfer only add a few additional functions to create annotated topic dataset.

The *create_bertopic* function works like Bertopic() and uses nearly the same arguments. It takes in entry parallel lists of texts (*docs*) and unique text identifiers (*ids*):

```python
from BERTransfer import create_bertopic

bertopic_model = create_bertopic(ids = ids, docs = docs, language = "english")
```

*create_bertopic* return an overlay object that contains the BERTopic model but also processed datasets that can be accessed through attributes. 

The datasets of topics that contain, for each topic, a list of characteristic words and characteristic documents. This is all the core information needed to perform the annotation of the topics.

```python
bertopic_model.topic_dataset
```

The dataset of documents include for each document the most likely topic and their associated probability. Associating the document with more detailed metadata may also help to identify relevant trends for the annotation (for instance the exclusive association of a topic to a specific event in a corpus based on social network.

```python
bertopic_model.document_dataset
```

Finally, *bertopic_model* also contains the embeddings of the topic, or basically their semantic signature within BERT. The embeddings will be instrumental to perform the transfer of topics from one corpus to another.

```python
bertopic_model.topic_embeddings
```

All these elements will be saved on the local directory using this command. The  project_name will be root name for all the subsequent files.

```python
bertopic_model.save_results(project_name = "twitter_2022_october")
```

## Topic transfer

Topic transfer is significantly quicker than topic modeling. The embeddings produced by sentence transformer will be used without the additional data processing of BERTopic. This makes BERTransfer a possible solution when dealing with very large corpora.

*create_bertransfer* works similarly to *create_bertopic*. The most important different is that we also pass a set of topic embeddings.

```python
from BERTransfer import create_bertransfer
bertransfer_model = create_bertransfer(ids, docs, topic_embeddings = topic_embeddings, language = "english", min_cosine_distance = 0.5)
```

So along with the ids and docs list of the new corpus it's necessary to open the topic embeddings of the previous corpus:

```python
import pickle
a_file = open("topic_embeddings_file.pkl", "rb")
topic_embeddings = pickle.load(a_file)
```

There's also a new indicators *min_cosine_distance* which is basically the threshold of proximity needed to associate a document to a topic. You can also think of it as a likelyhood of the document to belong to a topic. 0.5 is a fairly low requirement and to ensure better results you can pass a higher threshold: this would also result in having a larger share of the new database not being classified.

*create_bertransfer* return a new object. The document dataset associates each document to its closer topic in the semantic space, with cosine distance acting as a metric of likelyhood.

```python
bertransfer_model.document_dataset
```
There is not topic dataset as it is the same as the previous one.

It's also possible to reconcile the classification on this topic dataset to have interpretable names for topic with the document topic function.

```python
bertransfer_model.document_topic("topics_twitter_2022_october.tsv")
```

This function is also very practical when you have annotated topics based on the previous corpora: your annotations will appear as well.

```python
bertransfer_model.document_topic("topics_twitter_2022_october_documented.tsv").dropna()
```
