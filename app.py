# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import plotly.express as px
from textblob import TextBlob
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib
from pyspark.ml.feature import Tokenizer, NGram
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, length, udf,array
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import  StringType, ArrayType
from pyspark.sql import functions as F

# Setup Streamlit
st.set_page_config(page_title="Web Mining & Text Analysis", layout="wide")


# Setup SparkSession
@st.cache_resource
def create_spark():
    return SparkSession.builder.master("local[*]").appName("Text Mining App").getOrCreate()

spark = create_spark()

# Load spaCy Model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Fonction Wordcloud
def plot_wordcloud(text, title):
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# Chargement données
st.title("🚀 Web Mining & Text Mining App")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    chemin_local = uploaded_file.name
    with open(chemin_local, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = spark.read.option("header", "true").csv(chemin_local)
    df_pd = df.toPandas()

    st.subheader("Aperçu du Dataset")
    st.dataframe(df_pd.head())

    # Menu latéral Streamlit
    menu = ["Accueil", "Wordcloud & Clustering", "Système de Recommandation", "NER", "Analyse de Sentiment", "Traitement du texte", "Prédiction manuelle", "WordCloud par Sentiment","NGram","Prediction avec LogReg","Prediction avec XGBoost"]

    choice = st.sidebar.radio("Sélectionner une option", menu)

    if choice == "Accueil":
        st.subheader("Bienvenue sur l'application Web Mining et Text Mining !")
        st.write("Cette application vous permet d'analyser vos données textuelles, de créer des recommandations et d'extraire des insights.")
    
    elif choice == "Wordcloud & Clustering":
        st.header("📚 Text Mining : Wordcloud, LDA, KMeans")
        texts = df_pd['reviews.text'].fillna("").astype(str)

        # Wordcloud
        st.subheader("🔵 WordCloud Global")
        all_text = " ".join(texts)
        plot_wordcloud(all_text, "Wordcloud général")

        # Vectorisation
        cv = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
        tf = cv.fit_transform(texts)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(texts)

        n_topics = 10

        # Application du modèle LDA avec barre de progression
        st.subheader("🔄 Application du modèle LDA avec barre de progression")

        progress_bar = st.progress(0)
        status_text = st.empty()

        lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online')

        # Simulation d'une progression manuelle
        for percent_complete in range(0, 101, 10):
            status_text.text(f"Entraînement du modèle LDA... {percent_complete}% terminé")
            progress_bar.progress(percent_complete)
            # Pour simuler une charge (car fit est instantané normalement)
            import time
            time.sleep(0.1)

        # Maintenant fit réellement le modèle
        lda.fit(tf)
        lda_topics = lda.transform(tf)

        status_text.text("✅ Modèle LDA entraîné avec succès !")
        progress_bar.empty()


        kmeans = KMeans(n_clusters=n_topics, n_init=10).fit(tfidf)
        kmeans_labels = kmeans.predict(tfidf)

        svd = TruncatedSVD(n_components=50).fit(tfidf)
        reduced = svd.transform(tfidf)

        tsne = TSNE(n_components=2, perplexity=25, n_iter=1000)
        tsne_results = tsne.fit_transform(reduced)

        st.subheader("🎯 Top 15 mots par Topic (LDA)")
        feature_names = cv.get_feature_names_out()

        for idx, topic in enumerate(lda.components_):
            st.write(f"**Topic {idx}** :")
            st.write(", ".join([feature_names[i] for i in topic.argsort()[:-15 - 1:-1]]))

        st.subheader("🏷️ Top 15 mots par Cluster (KMeans)")
        df_tf = pd.DataFrame(tfidf.toarray())
        df_tf['cluster'] = kmeans_labels
        df_clustered = df_tf.groupby('cluster').sum().T

        for cluster_num in df_clustered.columns:
            top_words = df_clustered[cluster_num].nlargest(15).index.tolist()
            st.write(f"**Cluster {cluster_num}** :")
            st.write(", ".join([tfidf_vectorizer.get_feature_names_out()[i] for i in top_words]))

        st.subheader("🧩 Visualisation 2D (t-SNE)")
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cm.rainbow(np.linspace(0, 1, n_topics))

        for i in range(n_topics):
            idxs = np.where(kmeans_labels == i)
            ax.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=f'Cluster {i}', color=colors[i])

        plt.legend()
        plt.title("Clusters (KMeans) - t-SNE Reduction")
        st.pyplot(fig)

    elif choice == "Système de Recommandation":
        st.header("🎯 Système de Recommandation ALS")
        if st.button("Exécuter la recommandation ALS"):

            ratings = df.select(
                col("id").alias("reviewerID"),
                col("asins").alias("ElectroID"),
                col("`reviews.rating`").alias("rating")
            ).dropna()

            user_indexer = StringIndexer(inputCol="reviewerID", outputCol="userIndex")
            electro_indexer = StringIndexer(inputCol="ElectroID", outputCol="ElectroIndex")

            ratings = user_indexer.fit(ratings).transform(ratings)
            ratings = electro_indexer.fit(ratings).transform(ratings)

            ratings = ratings.select(
                col("userIndex").cast("int").alias("reviewerID"),
                col("ElectroIndex").cast("int").alias("ElectroID"),
                col("rating").cast("float")
            )

            # Supprimer les lignes où rating est NaN ou NULL
            ratings = ratings.filter(col("rating").isNotNull())

            # Split train/test (utile pour évaluation)
            train_data, test_data = ratings.randomSplit([0.8, 0.2], seed=42)

           # Charger le modèle déjà entraîné (changer le chemin vers le bon dossier)
            model = ALSModel.load("als_best_model")

            predictions = model.transform(test_data)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

            rmse = evaluator.evaluate(predictions)
            st.success(f"RMSE sur le jeu de test : {rmse:.3f}")

            userRecs = model.recommendForAllUsers(5)
            userRecs_pd = userRecs.limit(20).toPandas()

            st.dataframe(userRecs_pd)

            csv = userRecs_pd.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les recommandations CSV",
                data=csv,
                file_name='recommandations_als.csv',
                mime='text/csv'
            )

    elif choice == "NER":
        st.header("🧠 Extraction des Entités Nommées (NER)")
        if st.button("Exécuter NER"):

            @udf(StringType())
            def ner_udf(text):
                if text:
                    doc = nlp(text)
                    return str([(ent.text, ent.label_) for ent in doc.ents])
                else:
                    return None

            df_ner = df.withColumn("named_entities", ner_udf(col("`reviews.text`")))

            st.dataframe(df_ner.select("`reviews.text`", "named_entities").limit(50).toPandas())

    elif choice == "Analyse de Sentiment":
        st.header("💬 Analyse de Sentiment par Note")
        if st.button("Analyser les sentiments"):

            df_sentiment = df.withColumn("Sentiment",
                        when(col("`reviews.rating`") < 3, "Negative")
                        .when(col("`reviews.rating`") > 3, "Positive")
                        .otherwise("Neutral"))

            pandas_df = df_sentiment.toPandas()

            st.write("Distribution des Sentiments :")
            fig = px.histogram(pandas_df, x="Sentiment", color="Sentiment",
                               color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gold"})
            st.plotly_chart(fig)

            st.write("Distribution des Notes :")
            # Créez une figure et un axe
            fig, ax = plt.subplots()

            # Utilisez seaborn pour générer votre graphique
            sns.countplot(data=pandas_df, x="reviews.rating", ax=ax,order=[0, 1, 2, 3, 4, 5])

            # Affichez le graphique avec Streamlit
            st.pyplot(fig)
            # === Analyse Avancée par ASIN ===
            st.subheader("📦 Analyse des ASINs")

            # ASIN Frequency - normal et Log10
            fig_asin, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
            pandas_df["asins"].value_counts().plot(kind="bar", ax=ax1, title="ASIN Frequency")
            np.log10(pandas_df["asins"].value_counts()).plot(kind="bar", ax=ax2, title="ASIN Frequency (Log10 Adjusted)")
            st.pyplot(fig_asin)

            # Reviews.rating / ASINs
            st.subheader("Évolution de reviews.rating par ASIN")
            fig_rating, axes_rating = plt.subplots(2, 1, figsize=(16, 12))
            pandas_df["asins"].value_counts().plot(kind="bar", ax=axes_rating[0], title="ASIN Frequency")
            sns.pointplot(x="asins", y="reviews.rating", order=pandas_df["asins"].value_counts().index, data=pandas_df, ax=axes_rating[1])
            axes_rating[1].tick_params(axis='x', rotation=90)
            st.pyplot(fig_rating)


    elif choice == "Traitement du texte":
        st.header("📝 Prétraitement du texte (Tokenisation, Stopwords, Stemming)")
        
        # 1. Correction au départ
        df = df.withColumn("review_text", col("`reviews.text`"))
        df = df.filter(df["review_text"].isNotNull())

        # 2. Tokenizer pour découper le texte en mots
        tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
        df = tokenizer.transform(df)

        # 3. StopWordsRemover pour enlever les mots inutiles
        stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        df = stopwords_remover.transform(df)

        # 4. Stemming pour raciniser les mots
        stemmer = PorterStemmer()
        # Remplacer NULL par un array vide explicitement
        df = df.withColumn("filtered_words", when(col("filtered_words").isNull(), array().cast("array<string>")).otherwise(col("filtered_words")))
        stemming_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
        df = df.withColumn("stemmed", stemming_udf(col("filtered_words")))

        # 5. Affichage final
        st.dataframe(df.select("review_text", "filtered_words", "stemmed").toPandas())

    elif choice == "Prédiction manuelle":
        st.header("🔮 Prédiction manuelle des sentiments")
        text_input = st.text_area("Entrez votre texte ici :", "")
        if text_input:
            sentiment = TextBlob(text_input).sentiment.polarity
            if sentiment > 0:
                st.success(f"Sentiment positif (Polarity: {sentiment:.2f})")
            elif sentiment < 0:
                st.error(f"Sentiment négatif (Polarity: {sentiment:.2f})")
            else:
                st.warning(f"Sentiment neutre (Polarity: {sentiment:.2f})")
    elif choice == "WordCloud par Sentiment":
      st.header("🌈 WordCloud par Sentiment")

      if st.button("Générer les WordClouds"):
          df_sentiment = df.withColumn("Sentiment",
                      when(col("`reviews.rating`") < 3, "Negative")
                      .when(col("`reviews.rating`") > 3, "Positive")
                      .otherwise("Neutral"))

          df_sentiment_pd = df_sentiment.select("`reviews.text`", "Sentiment").toPandas()
          
          # On génère un WordCloud pour chaque catégorie de sentiment
          sentiments = ["Positive", "Negative", "Neutral"]

          for sentiment in sentiments:
              st.subheader(f"WordCloud pour les avis {sentiment}")
              text = " ".join(df_sentiment_pd[df_sentiment_pd["Sentiment"] == sentiment]["reviews.text"].dropna().astype(str))
              if text.strip() != "":
                  wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
                  fig, ax = plt.subplots(figsize=(10, 5))
                  ax.imshow(wordcloud, interpolation="bilinear")
                  ax.axis("off")
                  st.pyplot(fig)
              else:
                  st.info(f"Pas assez de textes pour {sentiment}")

    elif choice == "NGram":
      st.header("🔢 Analyse des N-grams avec PySpark")

      # Demander à l'utilisateur combien de n-grams il souhaite générer
      ngram_range = st.slider("Sélectionnez la taille des n-grams", 1, 3, 2)
      
      # 1. Correction au départ
      df = df.withColumn("review_text", col("`reviews.text`"))
      df = df.filter(df["review_text"].isNotNull())

      # Tokeniser les textes
      tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
      df_words = tokenizer.transform(df)
      
      # Générer des n-grams
      ngram = NGram(n=ngram_range, inputCol="words", outputCol="ngrams")
      df_ngrams = ngram.transform(df_words)

      # Afficher les n-grams les plus fréquents
      ngram_df = df_ngrams.select(F.explode(F.col("ngrams")).alias("ngram"))
      ngram_counts = ngram_df.groupBy("ngram").count().orderBy("count", ascending=False)

      # Convertir en DataFrame Pandas pour l'affichage dans Streamlit

      # Préparer les données pour la création du nuage de mots
      ngram_counts_pd = ngram_counts.limit(100).toPandas()  # Limiter aux 100 premiers n-grams
      st.subheader(f"Les {ngram_range}-grams les plus fréquents")
      st.dataframe(ngram_counts_pd)  # Afficher les n-grams sous forme de tableau


      # Créer un dictionnaire des n-grams et de leur fréquence
      ngram_freq = dict(zip(ngram_counts_pd['ngram'].astype(str), ngram_counts_pd['count']))

      # Générer le nuage de mots à partir des n-grams
      wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_freq)

      # Afficher le nuage de mots
      st.subheader(f"Nuage de mots des {ngram_range}-grams les plus fréquents")
      plt.figure(figsize=(10, 6))
      plt.imshow(wordcloud, interpolation="bilinear")
      plt.axis("off")
      st.pyplot(plt)

      # Visualisation des n-grams les plus fréquents sous forme de graphique à barres
      st.subheader("🔍 Visualisation des N-Grams")
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.barplot(x='count', y='ngram', data=ngram_counts_pd, ax=ax)
      st.pyplot(fig)
    elif choice == "Prediction avec LogReg":
        st.title("Prédiction de sentiment avec Logistic Regression")

        df = df.withColumn("review_text", col("`reviews.text`"))
        df_sentiment = df.withColumn("Sentiment",
                        when(col("`reviews.rating`") < 3, "Negative")
                        .when(col("`reviews.rating`") > 3, "Positive")
                        .otherwise("Neutral"))
        
        pandas_df = df_sentiment.toPandas()
        pandas_df = pandas_df.dropna(subset=['review_text'])

        try:
            lr_model = joblib.load("fichiers/lr_model.pkl")
            tfidf_m = joblib.load("fichiers/tfidf_vectorizer.pkl")
            label_encoder = joblib.load("fichiers/label_encoder.pkl")
            cat = label_encoder.classes_
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")
            st.stop()

        # Préparation des données pour évaluer les métriques
        X = tfidf_m.transform(pandas_df['review_text'])
        y = label_encoder.transform(pandas_df['Sentiment'])

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fonction de prédiction pour l'interface
        def predict_text(text):
            X_new = tfidf_m.transform([text])
            proba = lr_model.predict_proba(X_new)[0]
            return {cat[i]: proba[i] for i in range(len(cat))}

        # Interface utilisateur
        user_input = st.text_area("Entrez un texte à analyser")

        if st.button("Prédire"):
            if user_input.strip() == "":
                st.warning("Veuillez entrer un texte.")
            else:
                prediction = predict_text(user_input)
                st.subheader("Résultat de la prédiction :")
                for label, prob in prediction.items():
                    st.metric(label=f"{label}", value=f"{prob:.2%}")

        if st.checkbox("Afficher quelques exemples de reviews"):
            st.write(pandas_df[['review_text', 'Sentiment']].sample(5))

        # Calcul des métriques sur le test set
        y_pred = lr_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=cat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')

        # Affichage des métriques
        st.subheader("Rapport de classification sur les données de test :")
        st.text(report)
        st.write(f"Accuracy: {acc:.3%}")
        st.write(f"F1 score: {f1:.3f}")
    elif choice == "Prediction avec XGBoost":
        st.title("Prédiction de sentiment avec XGBoost (modèle chargé)")

        # Chargement modèle + TF-IDF + encodage
        boost = xgb.Booster()
        boost.load_model("fichiers/xgboost_sentiment_model.json")  # ← modèle sauvegardé
        tfidf_m = joblib.load("fichiers/tfidf_vectorizer.pkl")  # ← vectorizer sauvegardé
        label_encoder = joblib.load("fichiers/label_encoder.pkl")  # ← label encoder sauvegardé
        cat = label_encoder.classes_

        # Préparation DataFrame Spark
        df = df.withColumn("review_text", col("`reviews.text`"))
        df_sentiment = df.withColumn("Sentiment",
                        when(col("`reviews.rating`") < 3, "Negative")
                        .when(col("`reviews.rating`") > 3, "Positive")
                        .otherwise("Neutral"))

        pandas_df = df_sentiment.toPandas()
        pandas_df = pandas_df.dropna(subset=['review_text'])
        pandas_df['Sentiment_encoded'] = label_encoder.transform(pandas_df['Sentiment'])

        # TF-IDF vectorisation
        X = tfidf_m.transform(pandas_df['review_text'])
        y = pandas_df['Sentiment_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        test_set = xgb.DMatrix(X_test, label=y_test)

        # Fonction prédiction texte
        def predict_text(text):
            X_new = tfidf_m.transform([text])
            dmatrix_new = xgb.DMatrix(X_new)
            proba = boost.predict(dmatrix_new)[0]
            return {cat[i]: proba[i] for i in range(len(cat))}

        # Interface utilisateur
        user_input = st.text_area("Entrez un texte à analyser")
        if st.button("Prédire"):
            if user_input.strip() == "":
                st.warning("Veuillez entrer un texte.")
            else:
                prediction = predict_text(user_input)
                st.subheader("Résultat de la prédiction :")
                for label, prob in prediction.items():
                    st.metric(label=f"{label}", value=f"{prob:.2%}")

        if st.checkbox("Afficher quelques exemples de reviews"):
            st.write(pandas_df[['review_text', 'Sentiment']].sample(5))

        # Rapport classification sur test
        y_test_pred_proba = boost.predict(test_set)
        y_test_pred = y_test_pred_proba.argmax(axis=1)
        report = classification_report(y_test, y_test_pred, target_names=cat)
        st.subheader("Rapport de classification sur les données de test :")
        st.text(report)

        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='micro')
        st.write(f"Accuracy: {acc:.3%}")
        st.write(f"F1 score: {f1:.3f}")
    elif choice == "Prediction avec SVM":
        st.title("Prédiction de sentiment avec SVM (modèle chargé)")

        # Chargement modèle + TF-IDF + encodage
        model_svm=joblib.load("fichiers/svm_model.pkl")  # ← modèle sauvegardé
        tfidf= joblib.load("fichiers/tfidf_vectorizer_svm.pkl")  # ← vectorizer sauvegardé
        label_encoder = joblib.load("fichiers/label_encoder_svm.pkl")  # ← label encoder sauvegardé
        cat = label_encoder.classes_

        # Préparation DataFrame Spark
        df = df.withColumn("review_text", col("`reviews.text`"))
        df_sentiment = df.withColumn("Sentiment",
                        when(col("`reviews.rating`") < 3, "Negative")
                        .when(col("`reviews.rating`") > 3, "Positive")
                        .otherwise("Neutral"))

        pandas_df = df_sentiment.toPandas()
        pandas_df = pandas_df.dropna(subset=['review_text'])
        pandas_df['Sentiment_encoded'] = label_encoder.transform(pandas_df['Sentiment'])

        # TF-IDF vectorisation
        X = tfidf.transform(pandas_df['review_text'])
        y = pandas_df['Sentiment_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fonction prédiction texte
        def predict_text(text):
            X_new = tfidf.transform([text])
            proba = model_svm.predict_proba(X_new)[0]  # tableau des probabilités
            return {cat[i]: proba[i] for i in range(len(cat))}

        # Interface utilisateur
        user_input = st.text_area("Entrez un texte à analyser")
        if st.button("Prédire"):
            if user_input.strip() == "":
                st.warning("Veuillez entrer un texte.")
            else:
                prediction = predict_text(user_input)
                st.subheader("Résultat de la prédiction :")
                for label, prob in prediction.items():
                    st.metric(label=f"{label}", value=f"{prob:.2%}")

        if st.checkbox("Afficher quelques exemples de reviews"):
            st.write(pandas_df[['review_text', 'Sentiment']].sample(5))

        # Rapport classification sur test
        y_pred = model_svm.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=cat)
        print("\n📈 Rapport de classification :")
        print(report)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        st.write(f"Accuracy: {acc:.3%}")
        st.write(f"F1 score: {f1:.3f}")