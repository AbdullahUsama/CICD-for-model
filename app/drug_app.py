import gradio as gr
import skops.io as sio

# pipe = sio.load("model/drug_pipeline.skops", trusted=True)

import gradio as gr
import skops.io as sio

pipe = sio.load(
    "model/drug_pipeline.skops",
    trusted=[
        "sklearn.pipeline.Pipeline",
        "sklearn.compose.ColumnTransformer",
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.impute.SimpleImputer",
        "sklearn.preprocessing.OrdinalEncoder",
        "sklearn.preprocessing.StandardScaler",
        "numpy.ndarray",
        "numpy.dtype",    # <--- This is the type that was explicitly flagged
    ]
)



def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.

    Args:
        age (int): Age of patient
        sex (str): Sex of patient 
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood

    Returns:
        str: Predicted drug label
    """
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]

    label = f"Predicted Drug: {predicted_drug}"
    return label


inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]



title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = ""


gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch(share=True)
