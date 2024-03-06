import tkinter as tk
from tkinter import filedialog, ttk
from ttkthemes import ThemedTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from PIL import Image, ImageTk
import os

root = ThemedTk(theme="equilux")
root.title("Ensemble Name_rider Predictor")
root.geometry("1080x720")
root.configure(bg="#003c5c")

data = None
original_categories = None
ensemble_model = None

def browse_file():
    global data, original_categories
    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    original_categories = data['Name_Car'].astype('category').cat.categories
    show_data(data)

def show_data(data):
    treeview.delete(*treeview.get_children())
    for index, row in data.iterrows():
        values = (row['Year'], row['Volume'], row['Weight'], row['CO2'], row['Door'], row['Name_Car'])
        treeview.insert("", "end", values=values)
    for col in columns:
        treeview.heading(col, text=col, anchor='center')
        treeview.column(col, anchor='center')

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = resize_image(img, (300, 300))
        img = ImageTk.PhotoImage(img)
        lbl_img.config(image=img)
        lbl_img.image = img

def resize_image(img, new_size):
    return img.resize(new_size, resample=Image.BICUBIC)

def train_all_models():
    global original_categories, data, ensemble_model
    
    original_categories = data['Name_Car'].astype('category').cat.categories
    data['Name_Car'] = data['Name_Car'].astype('category')
    data['Name_Car'] = data['Name_Car'].cat.codes
    X = data[['Year', 'Volume', 'Weight', 'CO2', 'Door']]
    y = data['Name_Car']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Train each model
    logistic_regression_model = LogisticRegression()
    naive_bayes_model = GaussianNB()
    svm_model = SVC()
    decision_tree_model = DecisionTreeClassifier()
    random_forest_model = RandomForestClassifier()
    mlp_model = MLPClassifier()
    adaboost_model = AdaBoostClassifier()
    
    models = [('logistic_regression', logistic_regression_model),
              ('naive_bayes', naive_bayes_model),
              ('svm', svm_model),
              ('decision_tree', decision_tree_model),
              ('random_forest', random_forest_model),
              ('mlp', mlp_model),
              ('adaboost', adaboost_model)]
    
    ensemble_model = VotingClassifier(estimators=models, voting='hard')
    ensemble_model.fit(X_train, y_train)
    
    new_data = {
        'Year': int(entry_Year.get()),
        'Volume': int(entry_Volume.get()),
        'Weight': int(entry_Weight.get()),
        'CO2': int(entry_CO2.get()),
        'Door': int(entry_Door.get())
    }
    new_data_df = pd.DataFrame([new_data])
    
    # Make individual predictions for each model
    predictions = {}
    for model_name, model in models:
        model.fit(X_train, y_train)
        prediction = model.predict(new_data_df)
        predicted_category = pd.Categorical.from_codes(prediction.astype(int), categories=original_categories)
        predictions[model_name] = predicted_category[0]
    
    # Make ensemble prediction
    ensemble_prediction = ensemble_model.predict(new_data_df)
    
    predicted_category = pd.Categorical.from_codes(ensemble_prediction.astype(int), categories=original_categories)
    result_var.set(f'Predicted Name_Car: {predicted_category[0]}')
    show_result_window(predictions, predicted_category[0])
    data['Name_Car'] = pd.Categorical.from_codes(data['Name_Car'], categories=original_categories)




def reset_ensemble():
    global ensemble_model, original_categories
    ensemble_model = None
    original_categories = None

def reset_inputs():
    entry_Volume.delete(0, tk.END)
    entry_Volume.delete(0, tk.END)
    entry_CO2.delete(0, tk.END)
    entry_Year.delete(0, tk.END)
    entry_Door.delete(0, tk.END)
    result_var.set("Predicted Name_rider: ")
    reset_ensemble()

def reset_image():
    lbl_img.config(image=None)
    lbl_img.image = None

def show_result_window(predictions, ensemble_prediction):
    result_window = tk.Toplevel(root)
    result_window.title("Predicted Name Car Result")

    result_label_text = f'Ensemble Prediction: {ensemble_prediction}\n\n'
    result_label_text += 'Individual Model Predictions:\n'
    for model_name, prediction in predictions.items():
        result_label_text += f'{model_name}: {prediction}\n'

    result_label = tk.Label(result_window, text=result_label_text, font=("Arial", 14))
    result_label.pack(pady=20)

    image_label = tk.Label(result_window, text="Image preview:")
    image_label.pack()

    image_widget = tk.Label(result_window)
    image_widget.pack()

    filename = f'{ensemble_prediction}.jpeg'
    if os.path.exists(filename):
        image = Image.open(filename)
        photo = ImageTk.PhotoImage(image)
        image_widget.config(image=photo)
        image_widget.image = photo
    else:
        result_label_text = f'Predicted_Name_Car: {ensemble_prediction}.jpeg not found!'
        result_label = tk.Label(result_window, text=result_label_text, font=("Arial", 14), fg="red")
        result_label.pack(pady=20)

    ok_button = tk.Button(result_window, text="OK", command=result_window.destroy, bg="#1E90FF", fg="white")
    ok_button.pack()



frm_input = tk.Frame(root, padx=10, pady=10, bg="#003c5c")
frm_input.pack()

canvas = tk.Canvas(root, height=600, width=500)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

columns = ['Year : ปี', 'Volume : cc', 'Weight : น้ำหนัก', 'CO2 : อัตตรากันปล่อยก็าซ/กรัม', 'Door : กี่ประตู', 'Name_Car']
treeview = ttk.Treeview(canvas, columns=columns, show='headings')

font_size = 15
style = ttk.Style()
style.configure("Treeview.Heading", font=(None, font_size))

for col in columns:
    treeview.heading(col, text=col)

treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frm_image_input = tk.Frame(root, padx=10, pady=10, bg="#003c5c")
frm_image_input.pack()

lbl_img = tk.Label(frm_image_input)
lbl_img.pack()

label_Year = tk.Label(root, text="Year:", font=("Arial", 12), bg="#ff98e6", fg="black")
label_Year.pack(pady=5)

entry_Year = tk.Entry(root)
entry_Year.pack(pady=5)

label_Volume = tk.Label(root, text="Volume:", font=("Arial", 12), bg="#84ceff", fg="black")
label_Volume.pack(pady=5)

entry_Volume = tk.Entry(root)
entry_Volume.pack(pady=5)

label_Weight = tk.Label(root, text="Weight:", font=("Arial", 12), bg="#84ceff", fg="black")
label_Weight.pack(pady=5)

entry_Weight = tk.Entry(root)
entry_Weight.pack(pady=5)

label_CO2 = tk.Label(root, text="CO2:", font=("Arial", 12), bg="#ff98e6", fg="black")
label_CO2.pack(pady=5)

entry_CO2 = tk.Entry(root)
entry_CO2.pack(pady=5)

label_Door = tk.Label(root, text="Door:", font=("Arial", 12), bg="#ff98e6", fg="black")
label_Door.pack(pady=5)

entry_Door = tk.Entry(root)
entry_Door.pack(pady=5)

btn_browse = tk.Button(frm_input, text="Browse CSV", command=browse_file, bg="#4CAF50", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_browse.pack(pady=10)

btn_browse_image = tk.Button(frm_image_input, text="Browse Image", command=browse_image, bg="#FFA500", fg="black", relief=tk.FLAT, font=("Arial", 12))
btn_browse_image.pack(pady=10)

btn_reset_image = tk.Button(frm_image_input, text="Reset Image", command=reset_image, bg="#FF0000", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_reset_image.pack(pady=10)

btn_train_all_models = tk.Button(root, text="Predicted Name_Car", command=train_all_models, bg="#1E90FF", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_train_all_models.pack(pady=10)

btn_reset = tk.Button(root, text="Reset Inputs", command=reset_inputs, bg="#FF0000", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_reset.pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.pack()

root.mainloop()