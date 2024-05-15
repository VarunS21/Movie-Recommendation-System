from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model.movie_recommendation_system import recommend_movies

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/movie', methods=['POST'])
def movie():
    data = request.json
    resp = recommend_movies(data['movie'], data['range'])
    return jsonify(resp)

if __name__ == '__main__':
    app.run()