import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle

# Load the model and columns
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

def predict_survival():
    try:
        # Get user input
        pclass = int(pclass_entry.get())
        age = float(age_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        fare = float(fare_entry.get())
        sex = sex_var.get()
        embarked = embarked_var.get()

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex, embarked]], columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])

        # One-hot encode the input data
        input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=columns, fill_value=0)

        # Predict survival
        prediction = model.predict(input_data)[0]

        # Display the result
        result = 'Survived' if prediction == 1 else 'Did not survive'
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# Create the GUI
root = tk.Tk()
root.title("Titanic Survival Prediction")

tk.Label(root, text="Passenger Class (1, 2, or 3):").grid(row=0, column=0)
pclass_entry = tk.Entry(root)
pclass_entry.grid(row=0, column=1)

tk.Label(root, text="Age:").grid(row=1, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=1, column=1)

tk.Label(root, text="Siblings/Spouses Aboard:").grid(row=2, column=0)
sibsp_entry = tk.Entry(root)
sibsp_entry.grid(row=2, column=1)

tk.Label(root, text="Parents/Children Aboard:").grid(row=3, column=0)
parch_entry = tk.Entry(root)
parch_entry.grid(row=3, column=1)

tk.Label(root, text="Fare:").grid(row=4, column=0)
fare_entry = tk.Entry(root)
fare_entry.grid(row=4, column=1)

tk.Label(root, text="Sex (male or female):").grid(row=5, column=0)
sex_var = tk.StringVar()
sex_var.set('male')
tk.OptionMenu(root, sex_var, 'male', 'female').grid(row=5, column=1)

tk.Label(root, text="Embarked (C, Q, or S):").grid(row=6, column=0)
embarked_var = tk.StringVar()
embarked_var.set('S')
tk.OptionMenu(root, embarked_var, 'C', 'Q', 'S').grid(row=6, column=1)

tk.Button(root, text="Predict", command=predict_survival).grid(row=7, columnspan=2)

root.mainloop()
