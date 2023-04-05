class BERTransferM:
    def __init__(self, ids = None, embeddings=None, topic_embeddings = None, min_cosine_distance = 0.5, max_documents_topics = 15000):
        self.ids = ids
        self.topic_embeddings = topic_embeddings
        self.embeddings = embeddings
        self.min_cosine_distance = min_cosine_distance
        self.max_documents_topics = max_documents_topics
        self.document_size = len(self.embeddings)
        self.topic_size = len(self.topic_embeddings)
        self.embedding_transfer()

    def embedding_transfer(self):
        from scipy.spatial import distance
        import numpy as np
        import pandas as pd 
        current_topic_list = []
        document_in_topic_list = []
        text_list = []
        cosine_distance_list = []


        #For each topic we retrieve the closest documents
        for current_topic in self.topic_embeddings:
          distances = distance.cdist([self.topic_embeddings[current_topic]], self.embeddings, "cosine")[0]
          index_document_topic = np.argsort(distances)

          cosine_distance = 1
          index_document = 0
          for document_in_topic in index_document_topic[0:self.max_documents_topics]:
            text = self.ids[document_in_topic]
            cosine_distance = 1-distances[document_in_topic]
            if(cosine_distance > self.min_cosine_distance):
              current_topic_list.append(str(current_topic))
              document_in_topic_list.append(str(document_in_topic))
              text_list.append(text) 
              cosine_distance_list.append(cosine_distance)

        df = pd.DataFrame(list(zip(current_topic_list, document_in_topic_list, text_list, cosine_distance_list)), columns =['topic_id', 'document_id', 'url', 'cosine_distance'])
        
        #We only keep the best topic candidate per document id
        idx = df.groupby(['document_id'])['cosine_distance'].transform(max) == df['cosine_distance']
        df = df[idx]
        self.document_dataset = df
    
    #Document transfer
    def document_topic(self, topic_document_id):
        import pandas as pd
        topic_document = pd.read_csv(topic_document_id, sep='\t')
        topic_document = topic_document.loc[:,~topic_document.columns.str.contains('doc|topic_size', case=False)]
        self.document_dataset['topic_id'] = self.document_dataset['topic_id'].astype(str).astype(int)
        result_dataset = topic_document.merge(self.document_dataset, on="topic_id")
        return result_dataset

    def __repr__(self):
        return f"A BERTransfer model with {self.topic_size} topics and {self.document_size} documents."
