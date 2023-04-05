# BERTransfer
<img src="https://raw.githubusercontent.com/Pclanglais/BERTransfer/main/BERTransfer.png" alt="Bertransfer logo"  width="200"/>

BERTransfer is a text mining application that make it possible to apply topics defined for one corpus to another corpus.

BERTransfer is built on top of BERTopic and is part of the same ecosystem of BERT-based tools for text classification. BERTopic is a topic modeling technique that leverages 🤗 transformers and a custom class-based TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. BERTransfer makes it possible to reuse topics defined by BERTopic on one corpus to further corpora. 

BERTransfer help create annotated classification models from BERTopic using the following workflow:
* Topics are defined from an initial corpus.
* Topics can be annotated based on several features: the characteristic words and a list of representative documents.
* Topic annotations can be transfered to a new corpus.

## Use cases

BERTransfer has been used effectively in following tasks:
* Continuous observation of new data, such as an online conversation
* Dealing with very large corpus. In its default setting, BERTopic takes a long time on corpus larger than 70,000 documents. Topic transfer is a possible solution: topics have been successfully applied for corpus of millions of documents.
* Reuse complex annotations that may require the interpretation of an expert analyst.
