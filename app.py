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
from pyspark.sql.functions import col, sum, when, length, udf, array
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import functions as F
import os
from PIL import Image
import pickle
# Setup Streamlit
st.set_page_config(
    page_title="Web Mining & Text Analysis",
    layout="wide",
    page_icon=":mag:"
)

def load_css():
    # Chemin vers votre fichier CSS
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    
    # Vérifie si le fichier existe
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Fichier CSS non trouvé, utilisation des styles par défaut")

# Appelez cette fonction après st.set_page_config()
load_css()

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
                         height=1000,
                         colormap='viridis').generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# Chargement données
st.title("🚀 Web Mining & Text Mining Dashboard")
st.markdown("""
    <div style="background-color:#e9f7ef;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h3 style="color:#2c3e50;">Analysez vos données textuelles et découvrez des insights précieux</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar avec image logo
with st.sidebar:
    st.image("images/image.png", width=100)
    st.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    chemin_local = uploaded_file.name
    with open(chemin_local, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = spark.read.option("header", "true").csv(chemin_local)
    df_pd = df.toPandas()

    st.success("✅ Fichier chargé avec succès !")
    st.subheader("📊 Aperçu du Dataset")
    st.dataframe(df_pd.head().style.background_gradient(cmap='Blues'))

    # Menu latéral Streamlit
    menu = [
        "🏠 Accueil", 
        "☁️ Wordcloud & Clustering", 
        "🎯 Système de Recommandation", 
        "🧠 Extraction d'Entités", 
        "😊 Analyse de Sentiment", 
        "✂️ Traitement du texte", 
        "🔮 Prédiction manuelle",
        "🌈 WordCloud par Sentiment",
        "🔢 Analyse N-Gram",
        "📊 Prédiction LogReg",
        "🌳 Prédiction XGBoost",
        "⚡ Prédiction SVM"
    ]

    choice = st.sidebar.radio("Sélectionnez une option", menu)

    if choice == "🏠 Accueil":
        st.subheader("Bienvenue sur l'application Web Mining et Text Mining !")
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
            <h3 style="color:#2c3e50;">Fonctionnalités principales :</h3>
            <ul>
                <li>Analyse de sentiment avancée</li>
                <li>Visualisation avec WordClouds</li>
                <li>Modélisation thématique (LDA)</li>
                <li>Système de recommandation</li>
                <li>Extraction d'entités nommées</li>
                <li>Prédiction en temps réel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://www.uplix.fr/app/uploads/2021/07/Google-NLP-front-1-1024x512.png", width=200)
            st.markdown("**Analyse textuelle avancée** avec NLP et machine learning")
        
        with col2:
            st.image("images/data-visualization.webp", width=200)
            st.markdown("**Visualisations interactives** pour explorer vos données")
        
        # Nouvelle section pour les contributeurs
        st.markdown("---")
        st.subheader("👥 Équipe du Projet")
        
        col_contrib1, col_contrib2 = st.columns(2)
        
        with col_contrib1:
            st.image("images/michel.jpeg", width=150, caption="Prénom NOM")
            st.markdown("""
            <div style="text-align:center;">
                <h4>Michel Sagesse Kolié</h4>
                <p>Data Science|IA & Big data</p>
                <p>École: ENSA Tetouan</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_contrib2:
            st.image("images/kinda.jpeg", width=150, caption="Prénom NOM")
            st.markdown("""
            <div style="text-align:center;">
                <h4>Abdoul Latif Kinda</h4>
                <p>Data Science|IA & Big data</p>
                <p>École:École: ENSA Tetouan</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#e9f7ef;padding:15px;border-radius:10px;margin-top:20px;">
            <p style="text-align:center;">Projet réalisé dans le cadre du cours de Web Mining & Text Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif choice == "☁️ Wordcloud & Clustering": 
        st.title("☁️ Wordcloud & Clustering")
        st.image("Images/cluster_wordcloud.png", width=100)

        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Cette section vous permet de visualiser vos données textuelles sous forme de nuages de mots,
            d'identifier des thèmes via LDA, et de regrouper vos documents en clusters avec KMeans.
            """)

        texts = df_pd['reviews.text'].fillna("").astype(str)

        # WordCloud
        st.subheader("🔵 WordCloud Global")
        with st.spinner("Génération du WordCloud..."):
            all_text = " ".join(texts)
            plot_wordcloud(all_text, "Wordcloud général")

        # Paramètres utilisateur
        st.sidebar.subheader("Paramètres d'analyse")
        n_topics = st.sidebar.slider("Nombre de topics/clusters", 3, 15, 8)
        max_features = st.sidebar.slider("Nombre max de features", 1000, 5000, 2000)

        try:
            # Chargement des modèles
            with st.spinner("Chargement des modèles sauvegardés..."):
                models_dir = "fichiers"  # Dossier où sont stockés les modèles

                # Chargement via joblib
                cv = joblib.load(os.path.join(models_dir, "count_vectorizer.joblib"))
                idf_m = joblib.load(os.path.join(models_dir, "tfidf.joblib"))
                lda_m = joblib.load(os.path.join(models_dir, "lda_model.joblib"))
                kmean_m = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
                svd_v = joblib.load(os.path.join(models_dir, "svd_model.joblib"))

                # Chargement via pickle
                with open(os.path.join(models_dir, "tsne_model.pkl"), "rb") as f:
                    tnse_v = pickle.load(f)

            st.success("✅ Modèles chargés avec succès !")

            # Transformation
            tf = cv.transform(texts)
            tfidf = idf_m.transform(texts)

            # LDA
            st.subheader("🔄 Modélisation thématique (LDA)")
            lda_topics = lda_m.transform(tf)

            st.subheader(f"🎯 Top 15 mots par Topic (LDA - {lda_m.n_components} topics)")
            feature_names = cv.get_feature_names_out()

            cols = st.columns(2)
            for idx, topic in enumerate(lda_m.components_):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div style="background-color:#e9f7ef;padding:15px;border-radius:10px;margin-bottom:10px;">
                        <h4>Topic {idx}</h4>
                        <p>{", ".join([feature_names[i] for i in topic.argsort()[:-15 - 1:-1]])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Clustering KMeans
            st.subheader("🏷️ Clustering avec KMeans")
            kmeans_labels = kmean_m.predict(tfidf)

            # Visualisation t-SNE
            st.subheader("🧩 Visualisation des clusters (t-SNE)")
            with st.spinner("Réduction de dimension avec SVD et t-SNE..."):
                reduced = svd_v.transform(tfidf)
                tsne_results = tnse_v.fit_transform(reduced)

                fig, ax = plt.subplots(figsize=(12, 8))
                colors = cm.rainbow(np.linspace(0, 1, kmean_m.n_clusters))

                for i in range(kmean_m.n_clusters):
                    idxs = np.where(kmeans_labels == i)
                    ax.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], 
                            label=f'Cluster {i}', color=colors[i], alpha=0.6)

                plt.legend()
                plt.title("Visualisation des Clusters (KMeans) - t-SNE")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement ou de l'application des modèles : {e}")

    elif choice == "🎯 Système de Recommandation":
        st.header("🎯 Système de Recommandation ALS")
        st.image("images/ampoule.jpg", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Ce système de recommandation utilise l'algorithme ALS (Alternating Least Squares)
            pour suggérer des produits aux utilisateurs en fonction de leurs historiques d'évaluations.
            """)
        
        if st.button("🚀 Exécuter la recommandation ALS"):
            with st.spinner("Préparation des données..."):
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

                ratings = ratings.filter(col("rating").isNotNull())
                train_data, test_data = ratings.randomSplit([0.8, 0.2], seed=42)

            try:
                working_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(working_dir, "als_best_model")

                with st.spinner("Chargement du modèle ALS..."):
                    model = ALSModel.load(model_path)
                    predictions = model.transform(test_data)
                    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
                    rmse = evaluator.evaluate(predictions)

                st.success(f"✅ Modèle chargé avec succès (RMSE: {rmse:.3f})")
                
                # Générer et afficher les recommandations
                with st.spinner("Génération des recommandations..."):
                    userRecs = model.recommendForAllUsers(5)
                    userRecs_pd = userRecs.limit(20).toPandas()

                st.subheader("Top 20 utilisateurs avec leurs recommandations")
                st.dataframe(userRecs_pd.style.background_gradient(cmap='Greens'))

                csv = userRecs_pd.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les recommandations CSV",
                    data=csv,
                    file_name='recommandations_als.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Erreur lors du chargement ou de l'évaluation du modèle ALS : {e}")

    elif choice == "🧠 Extraction d'Entités":
        st.header("🧠 Extraction des Entités Nommées (NER)")
        st.image("images/Spacy_ner.jpg", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            L'extraction d'entités nommées identifie et classe les éléments clés dans le texte
            comme les personnes, organisations, lieux, etc. en utilisant spaCy.
            """)
        
        if st.button("🔍 Exécuter NER"):
            with st.spinner("Analyse des textes..."):
                @udf(StringType())
                def ner_udf(text):
                    if text:
                        doc = nlp(text)
                        return str([(ent.text, ent.label_) for ent in doc.ents])
                    else:
                        return None

                df_ner = df.withColumn("named_entities", ner_udf(col("`reviews.text`")))
                ner_sample = df_ner.select("`reviews.text`", "named_entities").limit(50).toPandas()

            st.subheader("Exemples d'entités extraites")
            st.dataframe(ner_sample.style.applymap(lambda x: 'background-color: #e6f3ff' if pd.notnull(x) else ''))
            
            # Exemple visuel
            example_text = "Apple is looking at buying U.K. startup for $1 billion"
            st.markdown("**Exemple d'analyse NER:**")
            st.code(f"""
            Texte: "{example_text}"
            Entités: [('Apple', 'ORG'), ('U.K.', 'GPE'), ('$1 billion', 'MONEY')]
            """, language='python')

    elif choice == "😊 Analyse de Sentiment":
        st.header("💬 Analyse de Sentiment")
        st.image("images/sentiment.png", width=100)

        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Analysez la polarité des avis en fonction des notes attribuées.
            Visualisez la distribution des sentiments et des notes.
            """)

        if st.button("📊 Analyser les sentiments"):
            with st.spinner("Calcul des sentiments..."):
                df_sentiment = df.withColumn("Sentiment",
                                when(col("`reviews.rating`") < 3, "Negative")
                                .when(col("`reviews.rating`") > 3, "Positive")
                                .otherwise("Neutral"))

                pandas_df = df_sentiment.toPandas()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribution des Sentiments")
                fig = px.pie(pandas_df, names="Sentiment", 
                            color="Sentiment",
                            color_discrete_map={"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#f39c12"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Distribution des Notes")
                fig = px.histogram(pandas_df, x="reviews.rating", nbins=6)
                st.plotly_chart(fig, use_container_width=True)

            # Analyse par produit
            st.subheader("📦 Analyse par Produit (ASIN)")

            asin_count = pandas_df["asins"].value_counts()
            top_asins = asin_count.nlargest(10).index

            tab1, tab2 = st.tabs(["Fréquence des ASINs", "Notes moyennes par ASIN"])

            with tab1:
                fig = px.bar(asin_count.nlargest(10), 
                            title="Top 10 ASINs par nombre d'avis")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Conversion de la colonne en numérique
                pandas_df['reviews.rating'] = pd.to_numeric(pandas_df['reviews.rating'], errors='coerce')

                # Puis le groupby et mean
                avg_rating = pandas_df.groupby("asins")["reviews.rating"].mean().loc[top_asins]
                fig = px.bar(avg_rating, title="Note moyenne des top 10 ASINs")
                st.plotly_chart(fig, use_container_width=True)


    elif choice == "✂️ Traitement du texte":
        st.header("📝 Prétraitement du texte")
        st.image("images/texte", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Cette section applique les étapes standard de NLP:
            - Tokenisation
            - Suppression des stopwords
            - Stemming
            """)
        
        # 1. Correction au départ
        df = df.withColumn("review_text", col("`reviews.text`"))
        df = df.filter(df["review_text"].isNotNull())

        # 2. Tokenizer pour découper le texte en mots
        with st.spinner("Tokenisation en cours..."):
            tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
            df = tokenizer.transform(df)

        # 3. StopWordsRemover pour enlever les mots inutiles
        with st.spinner("Suppression des stopwords..."):
            stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
            df = stopwords_remover.transform(df)

        # 4. Stemming pour raciniser les mots
        with st.spinner("Application du stemming..."):
            stemmer = PorterStemmer()
            df = df.withColumn("filtered_words", 
                             when(col("filtered_words").isNull(), 
                                  array().cast("array<string>")).otherwise(col("filtered_words")))
            stemming_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
            df = df.withColumn("stemmed", stemming_udf(col("filtered_words")))

        st.success("✅ Prétraitement terminé avec succès !")
        
        # Affichage des résultats
        st.subheader("Résultats du prétraitement")
        sample_size = st.slider("Nombre d'exemples à afficher", 5, 50, 10)
        processed_sample = df.select("review_text", "filtered_words", "stemmed").limit(sample_size).toPandas()
        
        st.dataframe(processed_sample.style.applymap(
            lambda x: 'background-color: #e6ffe6' if isinstance(x, list) and len(x) > 0 else ''
        ))

    elif choice == "🔮 Prédiction manuelle":
        st.header("🔮 Prédiction manuelle des sentiments")
        st.image("images/svm.png", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Analysez en temps réel la polarité d'un texte saisi.
            La prédiction utilise TextBlob pour une analyse simple mais efficace.
            """)
        
        text_input = st.text_area("📝 Entrez votre texte ici :", "", height=150)
        
        if st.button("🔍 Analyser le sentiment"):
            if text_input.strip() == "":
                st.warning("Veuillez entrer un texte à analyser.")
            else:
                with st.spinner("Analyse en cours..."):
                    sentiment = TextBlob(text_input).sentiment.polarity
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if sentiment > 0:
                        with col1:
                            st.metric(label="Sentiment", value="Positif", delta=f"Polarité: {sentiment:.2f}")
                            st.image("images/positif.avif", width=100)
                    elif sentiment < 0:
                        with col2:
                            st.metric(label="Sentiment", value="Négatif", delta=f"Polarité: {sentiment:.2f}")
                            st.image("images/negatif.jpg", width=100)
                    else:
                        with col3:
                            st.metric(label="Sentiment", value="Neutre", delta=f"Polarité: {sentiment:.2f}")
                            st.image("images/neutre.png", width=100)
                    
                    # Visualisation de la polarité
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(0, sentiment, color='#3498db' if sentiment > 0 else '#e74c3c')
                    ax.set_xlim(-1, 1)
                    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                    ax.set_title("Score de Polarité")
                    ax.axis('off')
                    st.pyplot(fig)

    elif choice == "🌈 WordCloud par Sentiment":
        st.header("🌈 WordCloud par Sentiment")
        st.image("images/wordcloud.png", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Visualisez les mots les plus fréquents pour chaque catégorie de sentiment.
            Comparez les mots clés des avis positifs, négatifs et neutres.
            """)
        
        if st.button("☁️ Générer les WordClouds"):
            with st.spinner("Calcul des WordClouds par sentiment..."):
                df_sentiment = df.withColumn("Sentiment",
                                when(col("`reviews.rating`") < 3, "Negative")
                                .when(col("`reviews.rating`") > 3, "Positive")
                                .otherwise("Neutral"))

                df_sentiment_pd = df_sentiment.select("`reviews.text`", "Sentiment").toPandas()
                
                sentiments = ["Positive", "Negative", "Neutral"]
                colors = ["Greens", "Reds", "Oranges"]
                
                tabs = st.tabs(sentiments)
                
                for i, sentiment in enumerate(sentiments):
                    with tabs[i]:
                        st.subheader(f"WordCloud {sentiment}")
                        text = " ".join(df_sentiment_pd[df_sentiment_pd["Sentiment"] == sentiment]["reviews.text"].dropna().astype(str))
                        
                        if text.strip() != "":
                            wordcloud = WordCloud(width=800, height=400, 
                                                background_color='white', 
                                                colormap=colors[i],
                                                stopwords=STOPWORDS).generate(text)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.set_title(f"Mots clés des avis {sentiment}", fontsize=16)
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.warning(f"Pas assez de textes pour {sentiment}")

    elif choice == "🔢 Analyse N-Gram":
        st.header("🔢 Analyse des N-Grams")
        st.image("images/n-grams.jpg", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Identifiez les séquences de mots les plus fréquentes (bigrammes, trigrammes).
            Utile pour détecter des expressions récurrentes dans vos données.
            """)
        
        ngram_range = st.slider("Sélectionnez la taille des n-grams", 1, 3, 2)
        
        if st.button("🔍 Analyser les N-Grams"):
            with st.spinner("Calcul des n-grams en cours..."):
                # 1. Préparation des données
                df = df.withColumn("review_text", col("`reviews.text`"))
                df = df.filter(df["review_text"].isNotNull())

                # Tokeniser les textes
                tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
                df_words = tokenizer.transform(df)
                
                # Générer des n-grams
                ngram = NGram(n=ngram_range, inputCol="words", outputCol="ngrams")
                df_ngrams = ngram.transform(df_words)

                # Compter les n-grams
                ngram_df = df_ngrams.select(F.explode(F.col("ngrams")).alias("ngram"))
                ngram_counts = ngram_df.groupBy("ngram").count().orderBy("count", ascending=False)
                ngram_counts_pd = ngram_counts.limit(50).toPandas()

            st.success(f"Top 50 {ngram_range}-grams identifiés !")
            
            # Affichage sous forme de tableau
            st.subheader(f"Top {ngram_range}-grams")
            st.dataframe(ngram_counts_pd.style.background_gradient(cmap='Blues'))
            
            # Visualisation
            tab1, tab2 = st.tabs(["Nuage de mots", "Graphique à barres"])
            
            with tab1:
                st.subheader(f"Nuage de mots des {ngram_range}-grams")
                ngram_freq = dict(zip(ngram_counts_pd['ngram'].astype(str), ngram_counts_pd['count']))
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    colormap='viridis').generate_from_frequencies(ngram_freq)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Top {ngram_range}-grams", fontsize=16)
                st.pyplot(fig)
            
            with tab2:
                st.subheader(f"Top 15 {ngram_range}-grams")
                top15 = ngram_counts_pd.head(15)
                fig = px.bar(top15, x='count', y='ngram', 
                            orientation='h',
                            color='count',
                            color_continuous_scale='Bluered')
                st.plotly_chart(fig, use_container_width=True)

    elif choice == "📊 Prédiction LogReg":
        st.title("📊 Prédiction de sentiment avec Logistic Regression")
        st.image("Images/logReg.png", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Ce modèle utilise une régression logistique entraînée pour prédire le sentiment.
            Il a été entraîné sur des milliers d'avis avec une vectorisation TF-IDF.
            """)
        
        # Préparation des données
        df = df.withColumn("review_text", col("`reviews.text`"))
        df_sentiment = df.withColumn("Sentiment",
                        when(col("`reviews.rating`") < 3, "Negative")
                        .when(col("`reviews.rating`") > 3, "Positive")
                        .otherwise("Neutral"))
        
        pandas_df = df_sentiment.toPandas()
        pandas_df = pandas_df.dropna(subset=['review_text'])

        try:
            # Chargement des modèles
            with st.spinner("Chargement des modèles..."):
                working_dir = os.path.dirname(os.path.abspath(__file__))
                lr_model_path = os.path.join(working_dir, "fichiers", "lr_model.pkl")
                tfidf_path = os.path.join(working_dir, "fichiers", "tfidf_vectorizer.pkl")
                label_enc_path = os.path.join(working_dir, "fichiers", "label_encoder.pkl")

                lr_model = joblib.load(lr_model_path)
                tfidf_m = joblib.load(tfidf_path)
                label_encoder = joblib.load(label_enc_path)
                cat = label_encoder.classes_

            st.success("✅ Modèles chargés avec succès !")

            # Interface de prédiction
            st.subheader("🔍 Testez le modèle")
            user_input = st.text_area("Entrez un texte à analyser", height=150)
            
            if st.button("Prédire le sentiment"):
                if user_input.strip() == "":
                    st.warning("Veuillez entrer un texte.")
                else:
                    with st.spinner("Analyse en cours..."):
                        X_new = tfidf_m.transform([user_input])
                        proba = lr_model.predict_proba(X_new)[0]
                        
                        # Affichage des résultats
                        cols = st.columns(3)
                        colors = ["#2ecc71", "#e74c3c", "#f39c12"]  # Vert, Rouge, Orange
                        
                        for i, (label, prob) in enumerate(zip(cat, proba)):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="metric" style="border-left: 5px solid {colors[i]}; padding-left: 10px;">
                                    <h3>{label}</h3>
                                    <h2>{prob:.1%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Graphique radar
                        fig = px.line_polar(
                            r=proba, 
                            theta=cat, 
                            line_close=True,
                            color_discrete_sequence=["#3498db"],
                            template="plotly_white"
                        )
                        fig.update_traces(fill='toself')
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            showlegend=False,
                            title="Probabilités de prédiction"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Métriques du modèle
            if st.checkbox("Afficher les performances du modèle"):
                st.subheader("📈 Performances du modèle")
                
                with st.spinner("Calcul des métriques..."):
                    X = tfidf_m.transform(pandas_df['review_text'])
                    y = label_encoder.transform(pandas_df['Sentiment'])
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    y_pred = lr_model.predict(X_test)
                    report = classification_report(y_test, y_pred, target_names=cat, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{acc:.1%}")
                    st.metric("F1 Score", f"{f1:.3f}")
                
                with col2:
                    st.dataframe(report_df.style.background_gradient(cmap='Blues'))
                
                # Matrice de confusion
                st.subheader("Matrice de confusion")
                conf_mat = pd.crosstab(
                    pd.Series(label_encoder.inverse_transform(y_test), name='Réel'),
                    pd.Series(label_encoder.inverse_transform(y_pred), name='Prédit')
                )
                
                fig = px.imshow(
                    conf_mat,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Prédit", y="Réel", color="Nombre")
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")


    elif choice == "🌳 Prédiction XGBoost":
        st.title("🌳 Prédiction avec XGBoost")
        st.image("images/XGBoost-Algorithm-2.webp", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Ce modèle utilise XGBoost, un algorithme de boosting d'arbres de décision,
            particulièrement efficace pour les problèmes de classification.
            """)
        
        try:
            # Chargement des modèles
            with st.spinner("Chargement des modèles XGBoost..."):
                working_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(working_dir, "fichiers", "xgboost_sentiment_model.json")
                tfidf_path = os.path.join(working_dir, "fichiers", "tfidf_vectorizer.pkl")
                label_enc_path = os.path.join(working_dir, "fichiers", "label_encoder.pkl")

                boost = xgb.Booster()
                boost.load_model(model_path)
                tfidf_m = joblib.load(tfidf_path)
                label_encoder = joblib.load(label_enc_path)
                cat = label_encoder.classes_

            st.success("✅ Modèles XGBoost chargés avec succès !")

            # Interface de prédiction
            st.subheader("🔍 Testez le modèle XGBoost")
            user_input = st.text_area("Entrez un texte à analyser (XGBoost)", height=150)
            
            if st.button("Prédire avec XGBoost"):
                if user_input.strip() == "":
                    st.warning("Veuillez entrer un texte.")
                else:
                    with st.spinner("Analyse en cours..."):
                        X_new = tfidf_m.transform([user_input])
                        dmatrix_new = xgb.DMatrix(X_new)
                        proba = boost.predict(dmatrix_new)[0]
                        
                        # Affichage des résultats
                        cols = st.columns(3)
                        colors = ["#2ecc71", "#e74c3c", "#f39c12"]  # Vert, Rouge, Orange
                        
                        for i, (label, prob) in enumerate(zip(cat, proba)):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="metric" style="border-left: 5px solid {colors[i]};">
                                    <h3>{label}</h3>
                                    <h2>{prob:.1%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Importance des features
                        st.subheader("Importance des features")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        xgb.plot_importance(boost, ax=ax, max_num_features=15)
                        st.pyplot(fig)

            # Métriques du modèle
            if st.checkbox("Afficher les performances du modèle XGBoost"):
                st.subheader("📈 Performances du modèle XGBoost")
                
                with st.spinner("Calcul des métriques..."):
                    # Préparation DataFrame
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
                    
                    y_test_pred_proba = boost.predict(test_set)
                    y_test_pred = y_test_pred_proba.argmax(axis=1)
                    
                    report = classification_report(y_test, y_test_pred, target_names=cat, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    acc = accuracy_score(y_test, y_test_pred)
                    f1 = f1_score(y_test, y_test_pred, average='weighted')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{acc:.1%}")
                    st.metric("F1 Score", f"{f1:.3f}")
                
                with col2:
                    st.dataframe(report_df.style.background_gradient(cmap='Greens'))
                
                # Courbe ROC
                st.subheader("Courbe ROC")
                fig, ax = plt.subplots(figsize=(8, 6))
                xgb.plot_importance(boost, ax=ax, max_num_features=15)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles XGBoost : {e}")

    elif choice == "⚡ Prédiction SVM":
        st.title("⚡ Prédiction avec SVM")
        st.image("images/svm.png", width=100)
        
        with st.expander("ℹ️ À propos de cette section"):
            st.write("""
            Ce modèle utilise un SVM (Support Vector Machine) linéaire,
            particulièrement efficace pour les problèmes de classification textuelle.
            """)
        
        try:
            # Chargement des modèles
            with st.spinner("Chargement des modèles SVM..."):
                working_dir = os.path.dirname(os.path.abspath(__file__))
                svm_model_path = os.path.join(working_dir, "fichiers", "svm_model.pkl")
                tfidf_path = os.path.join(working_dir, "fichiers", "tfidf_vectorizer_svm.pkl")
                label_enc_path = os.path.join(working_dir, "fichiers", "label_encoder_svm.pkl")

                model_svm = joblib.load(svm_model_path)
                tfidf = joblib.load(tfidf_path)
                label_encoder = joblib.load(label_enc_path)
                cat = label_encoder.classes_

            st.success("✅ Modèles SVM chargés avec succès !")

            # Interface de prédiction
            st.subheader("🔍 Testez le modèle SVM")
            user_input = st.text_area("Entrez un texte à analyser (SVM)", height=150)
            
            if st.button("Prédire avec SVM"):
                if user_input.strip() == "":
                    st.warning("Veuillez entrer un texte.")
                else:
                    with st.spinner("Analyse en cours..."):
                        X_new = tfidf.transform([user_input])
                        proba = model_svm.predict_proba(X_new)[0]
                        
                        # Affichage des résultats
                        cols = st.columns(3)
                        colors = ["#2ecc71", "#e74c3c", "#f39c12"]  # Vert, Rouge, Orange
                        
                        for i, (label, prob) in enumerate(zip(cat, proba)):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="metric" style="border-left: 5px solid {colors[i]};">
                                    <h3>{label}</h3>
                                    <h2>{prob:.1%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Graphique à barres
                        fig = px.bar(
                            x=cat, 
                            y=proba, 
                            color=cat,
                            color_discrete_sequence=colors,
                            labels={'x': 'Sentiment', 'y': 'Probabilité'},
                            title="Probabilités de prédiction"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Métriques du modèle
            if st.checkbox("Afficher les performances du modèle SVM"):
                st.subheader("📈 Performances du modèle SVM")
                
                with st.spinner("Calcul des métriques..."):
                    # Préparation DataFrame
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
                    
                    y_pred = model_svm.predict(X_test)
                    report = classification_report(y_test, y_pred, target_names=cat, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{acc:.1%}")
                    st.metric("F1 Score", f"{f1:.3f}")
                
                with col2:
                    st.dataframe(report_df.style.background_gradient(cmap='Oranges'))

        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles SVM : {e}")