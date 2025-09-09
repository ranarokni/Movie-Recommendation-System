# 🎬 Hybrid Movie Recommender System  

This project implements a **hybrid movie recommender** that combines **content-based filtering** with **collaborative filtering (SVD)** to generate personalized movie suggestions.  
The system was trained, evaluated, and finally deployed on **Hugging Face Spaces** using **Streamlit** as the interface. You can find the Hugging Face deployment [here](https://huggingface.co/spaces/ranarokni/Movie-Recommender)


---

## 🚀 Project Overview  

- **Content-based filtering**: Uses metadata such as keywords, cast, crew, genres, language, and descriptions (overview/tagline) to compute similarity between movies.  
- **Collaborative filtering**: Uses user–movie ratings and a **Surprise SVD** model to capture user preferences and predict unseen ratings.  
- **Hybrid reranking**: Combines both signals by first selecting content-similar movies and then reranking them with collaborative scores.  
- **Deployment**: Packaged as a Streamlit app on Hugging Face Spaces, allowing users to input a movie title and user ID to receive recommendations.  

Blending content with collaborative signals yields **more accurate, theme-consistent, and personalized recommendations** compared to either method alone.  

---

## 📂 Dataset  

The project uses the [**The Movies Dataset**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (MovieLens + TMDb):  
- ~45,000 movies released up to 2017  
- Metadata: cast, crew, genres, keywords, language, companies, collections, posters, release dates, votes  
- Ratings: ~26M ratings from 270k users (subset used: `ratings_small.csv` with 100k ratings from 700 users on ~9k movies)  

### Preprocessing Highlights  
- **Credits:** Selected only director + top 4 cast.  
- **Keywords:** Cleaned JSON, removed singletons, stemmed/standardized tokens.  
- **Metadata:** Normalized JSON fields (genres, companies, languages), handled nulls, and created `CleanMoviesMetadata.csv`.  
- **Feature Engineering:** Built a “soup” of tokens (keywords, cast, director, genres, language) for similarity scoring.  

---

## 📊 Exploratory Data Analysis (EDA)  

- **Genres:** Drama is the most prevalent (~18k titles).  
- **Budgets & Revenues:** Strong but noisy relationship; heavy-tailed distribution.  
- **Release Trends:** Sharp growth since 1930, dip around 2020 due to dataset truncation.  
- **Top Languages/Directors/Actors:** Clear dominance of English films and Hollywood-centered metadata.  
- **Distributions:** Action/Thriller show highest budgets and revenues, while Romance tends to be shorter and cheaper.  

---

## 🧠 Models  

### 1. **Baseline Model**  
- IMDB-style weighted ratings for genre-based top lists.  

### 2. **Collaborative Filtering**  
- **Algorithm:** Surprise SVD  
- **Training:** 5-fold CV (RMSE, MAE)  
- **Recommendation:** Predicts ratings for unseen movies, ranking top-K items per user.  

### 3. **Content-based Filtering**  
- **Text features:** TF-IDF on movie descriptions (overview + tagline).  
- **Metadata features:** Token soup (keywords, cast, director, genres, language).  
- **Similarity:** Cosine similarity via linear kernel.  

### 4. **Hybrid Model**  
- **Output:** Top-K recommendations personalized by both semantics and user history.  

### 5. **Deployment**  
- **Interface:** Streamlit app  
- **Hosting:** Hugging Face Spaces  
- **Features:**  
- Input 3 movie titles  
- Get hybrid recommendations with posters & metadata  
- Clean, reproducible end-to-end pipeline  

---


## ⚙️ Installation & Usage  

Clone the repo:  
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

#### Run locally with Streamlit:
```bash
streamlit run app/app.py
```

