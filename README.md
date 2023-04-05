# BERTransfer
BERTransfer is a text mining application that make it possible to transfer topics defined for one corpus to another corpus. The application is built on top of BERTopic and is part of the same ecosystem of BERT-based tools for text classification.

Topic transfer makes it possible to create annotated classification models from BERTopic that can be used in a semi-supervised ways:
* Topics are defined from an initial corpus.
* Topics can be annotated based on several features: the characteristic words and a list of 
* Topic annotations can be transfered to a new corpus.

This approach is especially useful in the following cases:
*For very large corpus. Currently BERTopic takes a long time on corpus larger than 70,000 documents.
*For continuous observation of new data.
*For the reuse complex annotations that may require the interpretation of an expert analyst.
