from flask import render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.linalg import eig
from app import app


def calculate_priority(matrix):
    eigvals, eigvecs = eig(matrix)
    max_eigval = np.max(eigvals)
    index = np.argmax(eigvals)
    priority_vector = np.real(eigvecs[:, index])
    priority_vector = priority_vector / np.sum(priority_vector)
    return priority_vector


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = pd.read_csv('data/raw_data.csv')
    mean_income = data['Annual Income (k$)'].mean()
    mean_spending = data['Spending Score (1-100)'].mean()

    criteria_matrix = np.array([
        [1, mean_income / mean_spending],
        [mean_spending / mean_income, 1]
    ])

    criteria_priority = calculate_priority(criteria_matrix)

    result = {
        "mean_income": mean_income,
        "mean_spending": mean_spending,
        "priority_vector": criteria_priority.tolist()
    }

    return jsonify(result)


@app.route('/data', methods=['GET'])
def data():
    data = pd.read_csv('data/raw_data.csv')
    return jsonify(data.to_dict(orient='records'))
