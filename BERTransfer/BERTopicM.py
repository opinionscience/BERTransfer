class BERTopicM:
    def __init__(self, topic_model = None, ids = None, topics=None, probs=None, embeddings=None, similarity_threshold = 0.01, document_selection = 20, add_meta_topics = False):
        self.topic_model = topic_model
        self.ids = ids
        self.topics = topics
        self.probs = probs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.document_selection = document_selection
        self.add_meta_topics = add_meta_topics
        self.document_size = len(self.embeddings)
        self.topic_size = len(list(set(self.topics)))

        #We create the names for the topics
        self.define_topic()

        #We create the dataset of documents
        self.create_document_dataset()

        #If required we include the meta topics
        if self.add_meta_topics:
          self.create_meta_topics()
        
        #We create the dataset of topics
        self.create_topic_dataset()

        #We retrieve the embeddings of the topics as dict (to be used for topic transfer)
        self.create_topic_embeddings()

    #A function to create the temporary names of the topic.
    def define_topic(self):
        list_topic = len(self.topic_model.get_topic_info())
        topic_word_3 = {}
        topic_word_10 = {}

        for current_topic in range(-1, list_topic-1):
          list_word = self.topic_model.get_topic(current_topic)
          word_10 = []
          word_3 = []

          id_word = 1
          try:
            for item in list_word:
              if id_word <= 3:
                word_3.append(item[0])
                word_10.append(item[0] + " (" + str(round(item[1]*100, 0)) + ")")
              else:
                word_10.append(item[0] + " (" + str(round(item[1]*100, 0)) + ")")
              id_word = id_word + 1
          except:
            print(current_topic)
            print(list_word)
          topic_word_3[current_topic] = "_".join(word_3)
          topic_word_10[current_topic] = ", ".join(word_10)

        self.topic_word_3 = topic_word_3
        self.topic_word_10 = topic_word_10
    
    #A dataset of each document based on the BERTopic model
    def create_document_dataset(self):
      import pandas as pd
      final_document = []
      final_prob = []
      final_topic = []
      final_topic_name_prob = []
      final_probtopic = []
      final_id = []

      id = 0
      for list_prob in self.probs:
        current_document = self.ids[id]
        current_topic = self.topics[id]
        prob_topic = -1
        for prob in list_prob:
          prob_topic = prob_topic + 1
          if prob > self.similarity_threshold: #The threshold of similarity
            final_document.append(current_document)
            final_topic.append(current_topic)
            final_prob.append(round(prob*100, 0))
            final_probtopic.append(prob_topic)
            final_topic_name_prob.append(self.topic_word_3[prob_topic])
            final_id.append(id+1)
        id = id+1

      documents = pd.DataFrame({"Id document": final_id, "Document": final_document, "Main Topic": final_topic, "Prob Topic": final_probtopic, "Prob Topic Name": final_topic_name_prob, "Prob": final_prob})

      idx = documents.groupby(['Document'])['Prob'].transform(max) == documents['Prob']
      documents = documents[idx]

      selector_d = {'Document': 'url', 'Prob Topic Name': 'topic_name', 'Prob Topic': 'topic', 'Prob': 'probability', 'Main Topic': 'main_topic'}
      self.document_dataset = documents.rename(columns=selector_d)[[*selector_d.values()]]

      #We also store the size of each topic
      self.documents_total = self.document_dataset.groupby(['topic']).size().reset_index(name='counts')
      self.documents_total_main = self.document_dataset.groupby(['main_topic']).size().reset_index(name='counts')
    
    #Optionally we can include a set of meta-topics.
    #This is a bit longer and not always necessary
    def create_meta_topics(self):
      from umap import UMAP
      import numpy as np
      from sklearn.cluster import DBSCAN
      from sklearn.preprocessing import MinMaxScaler
      from sklearn.preprocessing import StandardScaler

      # Extract topic words and their frequencies
      topics_set = sorted(list(self.topic_model.get_topics().keys()))

      # Embed c-TF-IDF into 2D
      all_topics = sorted(list(self.topic_model.get_topics().keys()))
      indices = np.array([all_topics.index(topic) for topic in topics_set])
      topic_embeddings = self.topic_model.c_tf_idf_.toarray()[indices]
      topic_embeddings = MinMaxScaler().fit_transform(topic_embeddings)
      topic_embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(topic_embeddings)

      # Extract the embeddings into a list
      final_embeddings = list(topic_embeddings)

      # Define the DBSCAN model
      dbscan = DBSCAN(eps=0.3, min_samples=3)

      # Fit the model to the data and predict the cluster labels
      cluster_labels = dbscan.fit(final_embeddings)

      self.meta_topics = cluster_labels.labels_

      self.topic_set_dict = {}

      topic_id = 0

      for meta_topic in cluster_labels.labels_.tolist():
        current_topic = str(topics_set[topic_id])
        self.topic_set_dict[current_topic] = meta_topic
        topic_id = topic_id +1
    
    #A function to create the dataset of topics to be annotated
    def create_topic_dataset(self):
      import pandas as pd
      topic_list_complete = []
      for topic_id in list(self.topic_word_3.keys()):
        if topic_id != -1:
          #Only if we have activated the meta topics.
          if self.add_meta_topics:
            meta_topic = self.topic_set_dict[str(topic_id)]
          topic_name = self.topic_word_3[topic_id]
          topic_words = self.topic_word_10[topic_id]
          try:
            topic_size = self.documents_total[self.documents_total['topic']==topic_id]['counts'].values.tolist()[0]
          except:
            topic_size = self.documents_total_main[self.documents_total_main['main_topic']==topic_id]['counts'].values.tolist()[0]

          try:
            topic_selection = self.document_dataset[self.document_dataset['topic']==topic_id].sort_values(by='probability', ascending=False).head(self.document_selection)
          except:
            topic_selection = self.document_dataset[self.document_dataset['main_topic']==topic_id].head(self.document_selection)

          list_url = topic_selection['url'].values.tolist()
          if self.add_meta_topics:
            topic_list_meta = [str(topic_id), topic_name, str(meta_topic), topic_words, str(topic_size)]
          else:
            topic_list_meta = [str(topic_id), topic_name, topic_words, str(topic_size)]
          topic_list_meta.extend(list_url)

          topic_list_complete.append(topic_list_meta)

      if self.add_meta_topics:
        col_final_list = ['topic_id', 'topic_name', 'meta_topic', 'topic_word', 'topic_size']
      else:
        col_final_list = ['topic_id', 'topic_name', 'topic_word', 'topic_size']

      for id_col in range(1, self.document_selection+1):
        id_col_name = 'doc_' + str(id_col)
        col_final_list.append(id_col_name)
      self.topic_dataset = pd.DataFrame(columns=col_final_list, data = topic_list_complete)
    
    def create_topic_embeddings(self):
      import numpy as np
      #We retrieve the list of topics
      self.topic_embeddings = {}

      id = 0
      for list_prob in self.probs:
        current_topic = str(self.topics[id])
        current_embedding = self.embeddings[id]
        if current_topic in self.topic_embeddings:
          self.topic_embeddings[current_topic] = np.add(self.topic_embeddings[current_topic], current_embedding)
        else:
          self.topic_embeddings[current_topic] = current_embedding
        id = id + 1
    
    #A function to save the output of BERTopic so that they could
    def save_results(self, project_name = "project"):
      #We save the document dataset
      self.document_dataset.to_csv('documents_' + project_name + '.tsv', sep = '\t', index=False)

      #We save the topic dataset
      self.topic_dataset.to_csv('topics_' + project_name + '.tsv', sep = '\t', index=False)

      #We save the embeddings
      import pickle
      a_file = open("topics_" + project_name + ".pkl", "wb")
      pickle.dump(self.topic_embeddings, a_file)
      a_file.close()
          
    def __repr__(self):
        return f"A BERTopic model with {self.topic_size} topics and {self.document_size} documents."
