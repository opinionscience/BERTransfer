#The function to create the BERTopic model.
#The main difference with the official BERTopic implementation is that we return also the document embeddings that will be necessary for further calculations
def create_bertopic(ids, docs, language = "english", calculate_probabilities=True, verbose=True, bert_model = "all-MiniLM-L6-v2", similarity_threshold = 0.01, document_selection = 20, add_meta_topics = True):
  from bertopic import BERTopic
  from sentence_transformers import SentenceTransformer
  from BERTransfer import BERTopicM

  sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = sentence_model.encode(docs, show_progress_bar=False)

  # Train our topic model using our pre-trained sentence-transformers embeddings
  topic_model = BERTopic(language=language, calculate_probabilities=True, verbose=True)
  topics, probs = topic_model.fit_transform(docs, embeddings)
  bertopic_model = BERTopicM(ids = ids, topic_model = topic_model, topics = topics, probs = probs, embeddings = embeddings, similarity_threshold = similarity_threshold, document_selection = document_selection, add_meta_topics = True)
  return bertopic_model

#The function to create apply the transfer of topics based on a previous dataset
def create_bertransfer(ids, docs, topic_embeddings, language = "english", bert_model = "all-MiniLM-L6-v2", min_cosine_distance = 0.5, max_documents_topics = 15000):
  from sentence_transformers import SentenceTransformer
  from BERTransfer import BERTransferM

  sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = sentence_model.encode(docs, show_progress_bar=True)
  bertransfer_model = BERTransferM(ids = ids, embeddings = embeddings, topic_embeddings = topic_embeddings, min_cosine_distance = 0.5, max_documents_topics = 15000)
  return bertransfer_model

#The function to load the topic embeddingsn as we store them with safe tensors.
#For now we provide a conversion to numpy.
def load_topic_embeddings(tensor_file):
  import torch
  from safetensors import safe_open
  from safetensors.torch import save_file
  topic_embeddings = {}
  with safe_open(tensor_file, framework="pt") as tensor_dict:
    for key in tensor_dict.keys():
      topic_embeddings[key] = tensor_dict.get_tensor(key).numpy()
