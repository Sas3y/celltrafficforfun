# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import joblib

# --- 1. SETUP ---
st.set_page_config(page_title="Cell Traffic Controller", page_icon="🧬")

# --- Model selection (SW vs TR) ---
# Allow switching model variant from the web UI. Options can be extended later.
with st.sidebar:
    st.subheader("Model Variant")
    variant = st.selectbox("Choose model artifact suffix:", options=["SW", "TR"], index=0)

# Build expected filenames based on variant
model_path = f'cell_traffic_model_{variant}.pkl'
features_path = f'model_features_{variant}.pkl'

def _try_load_paths(primary_model, primary_features):
    try:
        m = joblib.load(primary_model)
        fn = joblib.load(primary_features)
        return m, fn, primary_model, primary_features
    except FileNotFoundError:
        return None

# Try the selected-suffixed pair first
loaded = _try_load_paths(model_path, features_path)

if loaded is None:
    # Fallback: try the unsuffixed filenames produced by the notebook
    fallback = _try_load_paths('cell_traffic_model.pkl', 'model_features.pkl')
    if fallback is not None:
        model, feature_names, used_model_path, used_features_path = fallback
        st.warning(f"Using unsuffixed artifacts (`{used_model_path}`, `{used_features_path}`) as fallback. Consider renaming to include suffix '{variant}' for clarity.")
    else:
        st.error("🚨 Model files not found! Run the notebook first to generate .pkl files, or copy/rename the generated files to match the chosen variant.")
        st.info("Example PowerShell commands to create SW copies:\ncopy .\\cell_traffic_model.pkl .\\cell_traffic_model_SW.pkl\ncopy .\\model_features.pkl .\\model_features_SW.pkl")
        st.stop()
else:
    model, feature_names, used_model_path, used_features_path = loaded

# For debugging / visibility show which files were loaded
with st.sidebar:
    st.caption(f"Loaded model: {used_model_path}")
    st.caption(f"Loaded features: {used_features_path}")

# --- 2. LOGIC (Must match Notebook exactly) ---
hydro_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def extract_features_for_app(seq):
    seq = seq.upper().strip().replace('\n', '').replace(' ', '')
    if not seq: return None
    
    L = len(seq)
    feats = {}
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    
    # Global
    for aa in aa_list:
        feats[f"Global_{aa}"] = seq.count(aa) / L
        
    # N-Term
    n_term = seq[:50]
    feats["N_Term_Hydrophobicity"] = sum([hydro_scale.get(a,0) for a in n_term]) / len(n_term) if n_term else 0
    feats["N_Term_Positive_Charge"] = (n_term.count('K') + n_term.count('R')) / len(n_term) if n_term else 0
    feats["Global_Hydrophobicity"] = sum([hydro_scale.get(a,0) for a in seq]) / L
    
    # Ensure columns match training data order
    df = pd.DataFrame([feats])
    # Handle missing columns if any (safety) or reorder
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# --- 3. UI ---
st.title("🧬 The Cell Traffic Controller")
st.markdown("Predict subcellular localization using Random Forest.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Classes:**
    - 🧬 Nucleus
    - ⚡ Mitochondria
    - 📮 Secreted
    - 🧱 Cell Membrane
    - 💧 Cytoplasm
    """)
    st.caption("Powered by Scikit-Learn")

# Input
seq_input = st.text_area("Paste Protein Sequence", height=150)

if st.button("Predict", type="primary"):
    if seq_input:
        # Process
        X_input = extract_features_for_app(seq_input)
        
        # Predict
        pred_class = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]
        
        # Display
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            emoji = {"Mitochondria":"⚡", "Nucleus":"🧬", "Secreted":"📮", 
                     "Cytoplasm":"💧", "Cell Membrane":"🧱"}
            st.subheader(f"{emoji.get(pred_class, '')} {pred_class}")
            st.write("Prediction")
            
        with c2:
            # Chart
            probs_df = pd.DataFrame({
                'Location': model.classes_,
                'Confidence': pred_proba
            }).set_index('Location')
            st.bar_chart(probs_df)
            
        # Interpretation
        st.divider()
        st.caption("Feature Values:")
        cols = st.columns(3)
        cols[0].metric("Global Hydro", f"{X_input['Global_Hydrophobicity'][0]:.2f}")
        cols[1].metric("N-Term Hydro", f"{X_input['N_Term_Hydrophobicity'][0]:.2f}")
        cols[2].metric("N-Term Charge", f"{X_input['N_Term_Positive_Charge'][0]:.2f}")

    else:
        st.warning("Please enter a sequence.")