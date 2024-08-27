from flask import Flask, request, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Define the model path using os.path.join for cross-platform compatibility
model_path = os.path.join(os.path.dirname(
    __file__), 'models', 'model_data.pkl')

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
with open(model_path, 'rb') as file:
    model_data = pickle.load(file)

mlp_model = model_data['mlp_80_20']
vectorizer = model_data['vectorizer']


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    review = None
    if request.method == 'POST':
        review = request.form['review']
        review_vector = vectorizer.transform([review])
        prediction = mlp_model.predict(review_vector)[0]
    return render_template('index.html', review=review, prediction=prediction)


@app.route('/comparison')
def comparison():
    # Load CSV data
    csv_path = os.path.join(os.path.dirname(
        __file__), 'data', 'Final_Label_Preprocessed_Balance_Final-Plis.csv')
    df = pd.read_csv(csv_path, delimiter=';')

    # Select relevant columns
    df = df[['Product Name', 'Variations',
             'Rating', 'Date', 'Review', 'Label Manual']]

    # Vectorize the reviews
    reviews = df['Review'].tolist()
    review_vectors = vectorizer.transform(reviews)

    # Predict using the model
    df['Label Model'] = mlp_model.predict(review_vectors)

    # Compare manual and model labels
    df['Prediction Result'] = df['Label Manual'] == df['Label Model']
    x = df['Label Manual'] != df['Label Model']

    # Calculate the number of changes
    changes_count = x.sum()

    # Convert the DataFrame to HTML
    tables = df.to_html(classes='data', index=False)

    return render_template('comparison.html', tables=tables, changes_count=changes_count)


@app.route('/model_comparison')
def model_comparison():
    # Render a template that shows the comparison of models with different data proportions
    return render_template('model_comparison.html')


if __name__ == '__main__':
    app.run(debug=True)
