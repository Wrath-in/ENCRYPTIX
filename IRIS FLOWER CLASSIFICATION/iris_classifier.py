import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox

# Load the dataset
data = pd.read_csv('IRIS.csv')

# Encode the species column
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Define features and target
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Define a function to make predictions
def classify_iris():
    try:
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = clf.predict(input_data)
        species = label_encoder.inverse_transform(prediction)[0]
        
        messagebox.showinfo("Prediction", f"The predicted species is: {species}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values")

# Create the GUI
root = tk.Tk()
root.title("Iris Flower Classification")
root.geometry("400x300")

frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Sepal Length:").grid(column=0, row=0, sticky=tk.W)
entry_sepal_length = ttk.Entry(frame, width=10)
entry_sepal_length.grid(column=1, row=0, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Sepal Width:").grid(column=0, row=1, sticky=tk.W)
entry_sepal_width = ttk.Entry(frame, width=10)
entry_sepal_width.grid(column=1, row=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Petal Length:").grid(column=0, row=2, sticky=tk.W)
entry_petal_length = ttk.Entry(frame, width=10)
entry_petal_length.grid(column=1, row=2, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Petal Width:").grid(column=0, row=3, sticky=tk.W)
entry_petal_width = ttk.Entry(frame, width=10)
entry_petal_width.grid(column=1, row=3, sticky=(tk.W, tk.E))

ttk.Button(frame, text="Classify", command=classify_iris).grid(column=0, row=4, columnspan=2, pady=10)

for child in frame.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.mainloop()
