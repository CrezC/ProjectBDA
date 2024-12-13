import re
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


class RemarksAnalyzer():
    def __init__(self):
        """
        Initializes the RemarksAnalyzer class by setting up stop words for text preprocessing.
        """
        self.stop_words = set("""
            i me my myself we our ours ourselves you your yours yourself yourselves he him his
            himself she her hers herself it its itself they them their theirs themselves what
            which who whom this that these those am is are was were be been being have has had
            having do does did doing a an the and but if or because as until while of at by
            for with about against between into through during before after above below to
            from up down in out on off over under again further then once here there when
            where why how all any both each few more most other some such no nor not only own
            same so than too very s t can will just don should now d ll m o re ve y ain aren
            couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren
            won wouldn
        """.split())

    def load_reviews(self, file_path):
        """
        Loads review data from an Excel file.

        Args:
            file_path (str): Path to the Excel file containing reviews.

        Returns:
            DataFrame: A pandas DataFrame with review data.
        """
        reviews = pd.read_excel(file_path)
        reviews['Review'] = reviews['Review'].astype(str)
        return reviews

    def group_reviews_by_university(self, reviews):
        """
        Groups reviews by university.

        Args:
            reviews (DataFrame): DataFrame containing reviews.

        Returns:
            DataFrame: A DataFrame with reviews grouped by university.
        """
        return reviews.groupby('University')['Review'].apply(' '.join).reset_index()

    def preprocess_text(self, text):
        """
        Preprocesses a given text by removing punctuation, converting to lowercase, 
        and filtering out stopwords.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        words = text.split()
        filtered_words = [
            word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def preprocess_reviews(self, university_reviews):
        """
        Applies text preprocessing to all reviews in the DataFrame.

        Args:
            university_reviews (DataFrame): DataFrame containing reviews.

        Returns:
            DataFrame: DataFrame with preprocessed reviews.
        """
        university_reviews['Cleaned_Review'] = university_reviews['Review'].apply(
            self.preprocess_text)
        return university_reviews

    def topic_modeling(self, cleaned_reviews, num_topics=4):
        """
        Performs topic modeling using TF-IDF and NMF.

        Args:
            cleaned_reviews (DataFrame): DataFrame with preprocessed reviews.
            num_topics (int): Number of topics to extract.

        Returns:
            tuple: A DataFrame with topic probabilities and a dictionary of top words per topic.
        """
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            cleaned_reviews['Cleaned_Review'])
        nmf_model = NMF(n_components=num_topics, random_state=42)
        topic_matrix = nmf_model.fit_transform(tfidf_matrix)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        top_words_per_topic = {}
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            top_words_per_topic[f"Topic {topic_idx+1}"] = top_words

        topic_probabilities = pd.DataFrame(
            topic_matrix, columns=[f"Topic {i+1}" for i in range(num_topics)])
        return topic_probabilities, top_words_per_topic

    def clustering(self, topic_probabilities, num_clusters=4):
        """
        Performs KMeans clustering on topic probabilities.

        Args:
            topic_probabilities (DataFrame): DataFrame with topic probabilities.
            num_clusters (int): Number of clusters.

        Returns:
            Series: Cluster assignments for each review.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(topic_probabilities)
        return clusters

    def sentiment_analysis(self, reviews):
        """
        Analyzes sentiment of reviews and classifies them as Positive, Negative, or Neutral.

        Args:
            reviews (DataFrame): DataFrame containing reviews.

        Returns:
            DataFrame: DataFrame with sentiment scores and classifications.
        """
        reviews['Sentiment'] = reviews['Review'].apply(
            lambda r: TextBlob(r).sentiment.polarity)
        reviews['Sentiment_Class'] = reviews['Sentiment'].apply(
            lambda s: 'Positive' if s > 0.1 else (
                'Negative' if s < -0.1 else 'Neutral')
        )
        return reviews

    def calculate_similarity(self, positive_reviews, negative_reviews):
        """
        Calculates cosine similarity between positive and negative reviews.

        Args:
            positive_reviews (DataFrame): DataFrame with positive reviews.
            negative_reviews (DataFrame): DataFrame with negative reviews.

        Returns:
            DataFrame: Cosine similarity matrix.
        """
        tfidf_vectorizer = TfidfVectorizer()
        positive_tfidf = tfidf_vectorizer.fit_transform(
            positive_reviews['Cleaned_Review'])
        negative_tfidf = tfidf_vectorizer.transform(
            negative_reviews['Cleaned_Review'])
        similarity_matrix = cosine_similarity(positive_tfidf, negative_tfidf)
        return pd.DataFrame(
            similarity_matrix,
            index=positive_reviews['University'].values,
            columns=negative_reviews['University'].values
        )
    
    def centroid_sim(self, reviews):

        """
        Calculates the cosine similarity between cluster centroids of university reviews.

        Args:
            reviews (DataFrame): DataFrame containing reviews and their associated universities.
        
        Returns:
            DataFrame: A cosine similarity matrix for the cluster centroids.
        """
        reviews['Review'] = reviews['Review'].astype(str)
        university_reviews = reviews.groupby(
            'University')['Review'].apply(' '.join).reset_index()

        # Text preprocessing
        university_reviews['Cleaned_Review'] = university_reviews['Review'].apply(self.preprocess_text)

        # Perform TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(university_reviews['Cleaned_Review'])

        # Perform topic modeling
        num_topics = 5
        nmf_model = NMF(n_components=num_topics, random_state=42)
        topic_matrix = nmf_model.fit_transform(tfidf_matrix)
        topic_probabilities = pd.DataFrame(
            topic_matrix, columns=[f"Topic {i+1}" for i in range(num_topics)])
        university_reviews = pd.concat([university_reviews, topic_probabilities], axis=1)

        # Perform clustering
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        topic_clusters = kmeans.fit_predict(topic_probabilities)
        university_reviews['Topic_Cluster'] = topic_clusters

        # Calculate cluster centroids
        topic_columns = [col for col in university_reviews.columns if "Topic" in col and "Cluster" not in col]
        cluster_centroids = university_reviews.groupby('Topic_Cluster')[topic_columns].mean()

        # Calculate cosine similarity between centroids
        centroid_similarity_matrix = cosine_similarity(cluster_centroids)
        centroid_similarity_df = pd.DataFrame(
            centroid_similarity_matrix,
            index=[f"Cluster {i}" for i in range(cluster_centroids.shape[0])],
            columns=[f"Cluster {i}" for i in range(cluster_centroids.shape[0])]
        )
        return centroid_similarity_df


    def save_to_excel(self, df, output_path):
        """
        Saves a DataFrame to an Excel file.

        Args:
            df (DataFrame): DataFrame to save.
            output_path (str): Path to save the Excel file.
        """
        df.to_excel(output_path, index=False)

    def visualize_elbow_method(self, topic_probabilities):
        """
        Visualizes the Elbow Method to determine the optimal number of clusters.

        Args:
            topic_probabilities (DataFrame): DataFrame with topic probabilities.
        """
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=42)
            kmeanModel.fit(topic_probabilities)
            distortions.append(kmeanModel.inertia_)
        plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
