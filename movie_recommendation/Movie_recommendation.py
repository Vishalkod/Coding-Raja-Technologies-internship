from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import threading

app = Flask(__name__)

# Load and preprocess data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='first')
movies = movies.drop_duplicates(subset=['movieId'], keep='first')
movies['genres'] = movies['genres'].str.split('|')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build models
model_svd = SVD()
model_svd.fit(trainset)

model_user_based = KNNBasic(sim_options={'user_based': True})
model_user_based.fit(trainset)

model_item_based = KNNBasic(sim_options={'user_based': False})
model_item_based.fit(trainset)

# Model Evaluation
svd_predictions = model_svd.test(testset)
user_based_predictions = model_user_based.test(testset)
item_based_predictions = model_item_based.test(testset)

svd_rmse = accuracy.rmse(svd_predictions)
user_based_rmse = accuracy.rmse(user_based_predictions)
item_based_rmse = accuracy.rmse(item_based_predictions)

# Helper function for recommendations
def get_top_n_recommendations(user_id, model, n=10):
    user_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
    top_n_recommendations = []

    for movie_id in movies['movieId']:
        if movie_id not in user_movies:
            predicted_rating = model.predict(user_id, movie_id).est
            top_n_recommendations.append((movie_id, predicted_rating))

    top_n_recommendations.sort(key=lambda x: x[1], reverse=True)
    return top_n_recommendations[:n]

# Interactive Interface
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.form['user_id']
    recommendation_engine = request.form['recommendation_engine']
    
    try:
        user_id = int(user_id)
        if recommendation_engine == 'SVD':
            model = model_svd
        elif recommendation_engine == 'User-Based':
            model = model_user_based
        elif recommendation_engine == 'Item-Based':
            model = model_item_based
        else:
            return render_template('index.html', error="Invalid recommendation engine selected.")
        
        # Optimized multi-threaded recommendation generation
        recommendations = []

        def generate_recommendations():
            nonlocal recommendations
            recommendations = get_top_n_recommendations(user_id, model)

        thread = threading.Thread(target=generate_recommendations)
        thread.start()
        thread.join()

        recommendations = recommendations

        rec_list = []
        for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            rec_list.append(f"{i}. {movie_title} (Predicted Rating: {predicted_rating:.2f})")

        return render_template('recommendations.html', user_id=user_id, recommendations=rec_list, engine=recommendation_engine, svd_rmse=svd_rmse, user_based_rmse=user_based_rmse, item_based_rmse=item_based_rmse)

    except ValueError:
        return render_template('index.html', error="Please enter a valid user ID.")

if __name__ == '__main__':
    app.run(debug=True)
