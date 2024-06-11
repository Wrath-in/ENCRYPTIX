import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Load the dataset with specified encoding
try:
    data = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    messagebox.showerror("Error", "Failed to read the dataset due to encoding issues.")
    data = None

if data is not None:
    # Handle missing values by dropping rows with missing 'Rating'
    data = data.dropna(subset=['Rating'])
    
    # Handle missing values in other columns if necessary (here, we fill with a placeholder)
    data = data.fillna('Unknown')

    # Preprocess the data
    # Encode categorical features
    label_encoders = {}
    categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].astype(str))
        label_encoders[feature] = le

    # Define features and target
    X = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
    y = data['Rating']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'Root Mean Squared Error: {rmse}')

    # Create the GUI
    class MovieRatingPredictorApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Movie Rating Predictor")
            self.root.geometry("600x400")
            
            # Create GUI elements
            self.create_widgets()

        def create_widgets(self):
            # Title Label
            title_label = tk.Label(self.root, text="Movie Rating Predictor", font=("Helvetica", 16, "bold"))
            title_label.pack(pady=10)
            
            # Genre Entry
            self.genre_label = tk.Label(self.root, text="Genre:")
            self.genre_label.pack(pady=5)
            self.genre_entry = tk.Entry(self.root)
            self.genre_entry.pack(pady=5)

            # Director Entry
            self.director_label = tk.Label(self.root, text="Director:")
            self.director_label.pack(pady=5)
            self.director_entry = tk.Entry(self.root)
            self.director_entry.pack(pady=5)

            # Actor 1 Entry
            self.actor1_label = tk.Label(self.root, text="Actor 1:")
            self.actor1_label.pack(pady=5)
            self.actor1_entry = tk.Entry(self.root)
            self.actor1_entry.pack(pady=5)

            # Actor 2 Entry
            self.actor2_label = tk.Label(self.root, text="Actor 2:")
            self.actor2_label.pack(pady=5)
            self.actor2_entry = tk.Entry(self.root)
            self.actor2_entry.pack(pady=5)

            # Actor 3 Entry
            self.actor3_label = tk.Label(self.root, text="Actor 3:")
            self.actor3_label.pack(pady=5)
            self.actor3_entry = tk.Entry(self.root)
            self.actor3_entry.pack(pady=5)
            
            # Predict Button
            self.predict_button = tk.Button(self.root, text="Predict Rating", command=self.predict_rating)
            self.predict_button.pack(pady=20)
            
            # Result Label
            self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14))
            self.result_label.pack(pady=10)
        
        def predict_rating(self):
            try:
                # Get input values
                genre = self.genre_entry.get()
                director = self.director_entry.get()
                actor1 = self.actor1_entry.get()
                actor2 = self.actor2_entry.get()
                actor3 = self.actor3_entry.get()
                
                # Transform input values using label encoders
                genre = label_encoders['Genre'].transform([genre])[0]
                director = label_encoders['Director'].transform([director])[0]
                actor1 = label_encoders['Actor 1'].transform([actor1])[0]
                actor2 = label_encoders['Actor 2'].transform([actor2])[0]
                actor3 = label_encoders['Actor 3'].transform([actor3])[0]
                
                # Create input array
                input_data = np.array([[genre, director, actor1, actor2, actor3]])
                
                # Predict rating
                predicted_rating = model.predict(input_data)[0]
                self.result_label.config(text=f"Predicted Rating: {predicted_rating:.2f}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # Run the GUI application
    if __name__ == "__main__":
        root = tk.Tk()
        app = MovieRatingPredictorApp(root)
        root.mainloop()
else:
    print("Failed to load the dataset.")
