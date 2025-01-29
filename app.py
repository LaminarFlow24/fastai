from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect, send_file, url_for

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return render_template('options.html', filepath=filepath)


@app.route('/display_graphs', methods=['POST'])
def display_graphs():
    filepath = request.form['filepath']
    df = pd.read_csv(filepath)
    last_feature = df.columns[-1]

    graph_dir = 'static/graphs'
    os.makedirs(graph_dir, exist_ok=True)

    graph_paths = []
    for col in df.columns[:-1]:
        plt.figure()
        sns.scatterplot(data=df, x=col, y=last_feature)
        plt.title(f'{col} vs {last_feature}')
        plt.xlabel(col)
        plt.ylabel(last_feature)
        graph_path = os.path.join(graph_dir, f'{col}_vs_{last_feature}.png')
        plt.savefig(graph_path)
        plt.close()
        graph_paths.append(graph_path)

    graph_links = ''.join([f'<li><img src="/{path}" alt="{path}" style="max-width:500px;"/></li>' for path in graph_paths])

    return render_template('graphs.html', graph_links=graph_links)


import pickle
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import save_model

@app.route('/train_model', methods=['POST'])
def train_model():
    filepath = request.form['filepath']
    df = pd.read_csv(filepath)

    # Preprocess the data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    old_X = X

    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

    # Impute non-numeric columns with most frequent value
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    X[non_numeric_cols] = non_numeric_imputer.fit_transform(X[non_numeric_cols])

    # One-hot encode categorical features
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        transformed = encoder.fit_transform(X[[col]])
        encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([col]))
        X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
        encoders[col] = encoder

    # Normalize data
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the percentage of MAE compared to the mean of y_test
    mean_y = y_test.mean()
    mae_percentage = (mae / mean_y) * 100

    # Store preprocessing artifacts
    app.config['model'] = model
    app.config['scaler'] = scaler
    app.config['encoders'] = encoders
    app.config['numeric_imputer'] = numeric_imputer
    app.config['non_numeric_imputer'] = non_numeric_imputer
    app.config['y_min'] = y_min
    app.config['y_max'] = y_max
    app.config['X_columns'] = X.columns
    app.config['old_X_columns'] = old_X.columns

    # Save model, scaler, and encoders
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'linear_model.pkl')
    scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scaler.pkl')
    encoders_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoders.pkl')

    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save the encoders
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)

    input_fields = ''.join([f'<label>{col}: <input type="text" name="{col}"></label><br>' for col in df.columns[:-1]])

    return render_template('model_training.html', 
                       mae=mae, 
                       mean_y=mean_y, 
                       mae_percentage=mae_percentage, 
                       model_path=model_path, 
                       scaler_path=scaler_path, 
                       encoders_path=encoders_path, 
                       input_fields=input_fields)


@app.route('/download', methods=['GET'])
def download_file():
    file_path = request.args.get('file')
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found.", 404



@app.route('/predict', methods=['POST'])
def predict():
    model = app.config['model']
    scaler = app.config['scaler']
    encoders = app.config['encoders']
    y_min = app.config['y_min']
    y_max = app.config['y_max']
    X_columns = app.config['X_columns']
    old_X_columns = app.config['old_X_columns']

    input_data = {}

    for col in old_X_columns:
        # Handle categorical columns
        if col in encoders:
            # Extract the raw column name (e.g., 'mainroad' from 'mainroad_yes')
            raw_col_name = col
            raw_input = request.form.get(raw_col_name)

            if raw_input is None:
                return f"Missing input for column '{raw_col_name}'."

            try:
                # Transform the raw input using the encoder
                transformed = encoders[raw_col_name].transform([[raw_input]])
                encoded_df = pd.DataFrame(transformed, columns=encoders[raw_col_name].get_feature_names_out([raw_col_name]))

                # Add each one-hot-encoded value to input_data
                for encoded_col in encoded_df.columns:
                    input_data[encoded_col] = encoded_df[encoded_col].iloc[0]  # Scalar value

            except ValueError:
                return f"Invalid input for column '{raw_col_name}': '{raw_input}' is not valid."

        # Handle numeric columns
        else:
            try:
                input_data[col] = float(request.form[col])  # Assign scalar value, not a list
            except ValueError:
                return f"Invalid input for column '{col}': expected a numeric value."
            except KeyError:
                return f"Missing input for column '{col}'."



    # Convert input_data to DataFrame and reindex to match X_columns
    input_df = pd.DataFrame([input_data]).reindex(columns=X_columns, fill_value=0)

    # Debug: Print input_df to check its contents
    print("Input DataFrame before scaling:")
    print(input_df)

    # Normalize the input data
    try:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    except ValueError as e:
        print(f"Error during scaling: {e}")
        return f"Error during scaling: {e}"

    # Predict and unnormalize the output
    prediction = model.predict(input_df)[0]
    prediction = prediction * (y_max - y_min) + y_min

    return render_template('prediction.html', prediction=prediction)

@app.route('/train_random_forest', methods=['POST','GET'])
def train_random_forest():
    print("Form data received:", request.form)  # Debugging line ################################################################
    filepath = request.form.get('filepath', None)
    if not filepath:
        return "Filepath is required.", 400


    filepath = request.form['filepath']
    df = pd.read_csv(filepath)

    # Preprocess the data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    old_X = X

    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

    numeric_imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    X[non_numeric_cols] = non_numeric_imputer.fit_transform(X[non_numeric_cols])

    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        transformed = encoder.fit_transform(X[[col]])
        encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([col]))
        X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
        encoders[col] = encoder

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    from sklearn.ensemble import RandomForestRegressor
    hyperparams = app.config.get('HYPERPARAMETERS', {})
    model = RandomForestRegressor(
        n_estimators=hyperparams.get('n_estimators', 100),
        max_depth=hyperparams.get('max_depth', None),
        min_samples_split=hyperparams.get('min_samples_split', 2),
        max_leaf_nodes=hyperparams.get('max_leaf_nodes', None),
        min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
        max_features=hyperparams.get('max_features', None),
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mean_y = y_test.mean()
    mae_percentage = (mae / mean_y) * 100

    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'random_forest_model.pkl')
    scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scaler.pkl')
    encoders_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoders.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)

    input_fields = ''.join([f'<label>{col}: <input type="text" name="{col}"></label><br>' for col in df.columns[:-1]])

    return render_template('rf_model_training.html',
                            mae=mae,
                            mean_y=mean_y,
                            mae_percentage=mae_percentage,
                            model_path=model_path,
                            scaler_path=scaler_path,
                            encoders_path=encoders_path,
                            input_fields=input_fields)


@app.route('/rf_predict', methods=['POST'])
def rf_predict():
    model = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'], 'random_forest_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'], 'scaler.pkl'), 'rb'))
    encoders = pickle.load(open(os.path.join(app.config['UPLOAD_FOLDER'], 'encoders.pkl'), 'rb'))

    input_data = {}

    for col in app.config.get('old_X_columns', []):
        if col in encoders:
            raw_input = request.form.get(col)
            if raw_input is None:
                return f"Missing input for column '{col}'."
            try:
                transformed = encoders[col].transform([[raw_input]])
                encoded_df = pd.DataFrame(transformed, columns=encoders[col].get_feature_names_out([col]))
                for encoded_col in encoded_df.columns:
                    input_data[encoded_col] = encoded_df[encoded_col].iloc[0]
            except ValueError:
                return f"Invalid input for column '{col}'."
        else:
            try:
                input_data[col] = float(request.form[col])
            except (ValueError, KeyError):
                return f"Invalid input for column '{col}'."

    input_df = pd.DataFrame([input_data]).reindex(columns=app.config.get('X_columns', []), fill_value=0)

    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    prediction = model.predict(input_df)[0]
    y_min = app.config['y_min']
    y_max = app.config['y_max']
    prediction = prediction * (y_max - y_min) + y_min

    return render_template('prediction.html', prediction=prediction)


@app.route('/rf_hyperparameters', methods=['GET', 'POST'])
def rf_hyperparameters():
    if request.method == 'POST':
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth', None)
        max_depth = int(max_depth) if max_depth and max_depth.isdigit() else None
        min_samples_split = int(request.form.get('min_samples_split', 2))
        max_leaf_nodes = request.form.get('max_leaf_nodes', None)
        max_leaf_nodes = int(max_leaf_nodes) if max_leaf_nodes and max_leaf_nodes.isdigit() else None
        min_samples_leaf = int(request.form.get('min_samples_leaf', 1))
        max_features = request.form.get('max_features', None)

        hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
        
        app.config['HYPERPARAMETERS'] = hyperparameters
        
        return redirect('/train_random_forest')

    return render_template('rf_hyperparameters.html')






if __name__ == '__main__':
    app.run(debug=True)
