import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from tensorflow import keras

# -------------------------
# 1. Define target functions
# -------------------------

def f_poly(x):
    return x**3 - 4*x**2 + x - 5

def f_sin(x):
    return np.sin(x)

def f_exp(x):
    return np.exp(x)

def f_piecewise(x):
    return np.where(x < 0, x**2, x)

FUNC_DEFS = {
    "Polynomial: x^3 - 4x^2 + x - 5": ("poly", f_poly),
    "Sine: sin(x)": ("sin", f_sin),
    "Exponential: e^x": ("exp", f_exp),
    "Piecewise: x^2 (x<0), x (x>=0)": ("piecewise", f_piecewise),
}

ARCH_CONFIGS = [
    ("shallow_relu", "ReLU (shallow)"),
    ("medium_relu",  "ReLU (medium)"),
    ("deep_relu",    "ReLU (deep)"),
    ("deep_tanh",    "Tanh (deep)"),
]

MODELS_DIR = Path("models")  # directory containing your .h5 files


# -------------------------
# 2. Load models (cached)
# -------------------------

@st.cache_resource
def load_models():
    models = {}
    for func_short_name in ["poly", "sin", "exp", "piecewise"]:
        for arch_name, _ in ARCH_CONFIGS:
            filename = f"{func_short_name}_{arch_name}.h5"
            path = MODELS_DIR / filename
            if not path.exists():
                continue
            models[(func_short_name, arch_name)] = keras.models.load_model(
                path,
                compile=False  
            )
    return models

# Call it once at import time to create the global "models"
models = load_models()


# -------------------------
# 3. Streamlit UI
# -------------------------

st.title("ANN Function Approximation Demo")

st.sidebar.header("Settings")

func_label = st.sidebar.selectbox(
    "Select target function",
    list(FUNC_DEFS.keys())
)

func_short_name, func = FUNC_DEFS[func_label]

x_val = st.sidebar.slider(
    "Select x value",
    min_value=-5.0,
    max_value=5.0,
    value=0.0,
    step=0.1
)

st.write(f"### Selected function")
st.latex({
    "poly": r"f(x) = x^3 - 4x^2 + x - 5",
    "sin": r"f(x) = \sin(x)",
    "exp": r"f(x) = e^x",
    "piecewise": r"f(x) = \begin{cases} x^2, & x<0 \\ x, & x\ge0 \end{cases}"
}[func_short_name])

# True value
y_true = float(func(np.array([x_val])))

st.write(f"**x = {x_val:.3f}**")
st.write(f"**True f(x) = {y_true:.6f}**")

# -------------------------
# 4. Compare model outputs
# -------------------------

rows = []
for arch_name, arch_label in ARCH_CONFIGS:
    key = (func_short_name, arch_name)
    if key not in models:
        continue

    model = models[key]
    y_pred = float(model.predict(np.array([[x_val]]), verbose=0))
    abs_err = abs(y_pred - y_true)

    rows.append({
        "Architecture": arch_label,
        "Model name": arch_name,
        "Predicted f(x)": y_pred,
        "Absolute error": abs_err,
    })

if rows:
    df = pd.DataFrame(rows)
    st.write("### Model predictions at selected x")
    st.dataframe(df.style.format({"Predicted f(x)": "{:.6f}", "Absolute error": "{:.6f}"}))
else:
    st.error("No models loaded – please check that the 'models' folder is in the same directory as this script.")

# -------------------------
# 5. Plot true vs model curves
# -------------------------

st.write("### Function approximation over [-5, 5]")

x_grid = np.linspace(-5, 5, 400).reshape(-1, 1)
y_true_grid = func(x_grid)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_grid, y_true_grid, label="True f(x)", color="black", linewidth=2)

colors = {
    "shallow_relu": "tab:red",
    "medium_relu": "tab:blue",
    "deep_relu": "tab:green",
    "deep_tanh": "tab:orange",
}

for arch_name, arch_label in ARCH_CONFIGS:
    key = (func_short_name, arch_name)
    if key not in models:
        continue
    model = models[key]
    y_pred_grid = model.predict(x_grid, verbose=0)
    ax.plot(x_grid, y_pred_grid, label=arch_label, color=colors.get(arch_name, None), alpha=0.8)

ax.axvspan(-3, 3, color="gray", alpha=0.1, label="Training range [-3, 3]")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)

st.pyplot(fig)