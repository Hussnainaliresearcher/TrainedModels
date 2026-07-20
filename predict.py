# ==============================================================================
# PIPELINE PREDICTION APP
# Model: Gradient Boosting | Task: Classification
# ==============================================================================
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import boto3

st.set_page_config(page_title="Random Forest Predictor Model", layout="centered")

@st.cache_resource
def load_pipeline(pkl_path="pipeline.pkl"):
    """Loads the model pipeline. Priority is given to S3 storage if configured."""
    s3_bucket = os.getenv("AWS_S3_BUCKET")
    s3_model_key = os.getenv("AWS_S3_MODEL_KEY") # e.g., 'model.pkl' or 'pipeline.pkl'
    
    # If custom S3 key is provided, update target path name
    target_path = s3_model_key if s3_model_key else pkl_path

    if s3_bucket and s3_model_key:
        try:
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, s3_model_key, target_path)
            pkl_path = target_path
        except Exception as e:
            st.warning(f"S3 download failed ({e}). Attempting to read from local workspace...")

    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def preprocess_data(df_raw, pipeline):
    """Mimics the main app's preprocessing to convert human inputs into model features."""
    df = df_raw.copy()
    enc_meta = pipeline.get("enc_meta", {})
    scalers = pipeline.get("scalers", {})
    features = pipeline.get("feature_cols", [])
    
    # 1. Apply Categorical Encodings (Label / One-Hot)
    for col, meta in enc_meta.items():
        if col not in df.columns:
            continue
            
        method = meta.get("method")
        if method in ["label", "ordinal"]:
            stored_map = meta.get("stored_map", {})
            df[col] = df[col].map(stored_map).fillna(0) # Fallback to 0
            
        elif method == "onehot":
            dummy_cols = meta.get("stored_dummy_cols", [])
            for d_col in dummy_cols:
                prefix = f"{col}_"
                if d_col.startswith(prefix):
                    val = d_col[len(prefix):]
                    df[d_col] = (df[col].astype(str) == str(val)).astype(float)
                else:
                    df[d_col] = 0.0
            df = df.drop(columns=[col])
            
    # 2. Apply Numerical Scalers
    for col, scaler in scalers.items():
        if col in df.columns:
            try:
                df[col] = scaler.transform(df[[col]].values)
            except Exception:
                pass
                
    # 3. Ensure exact column match for model
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
            
    return df[features]

def decode_target(pred, pipeline):
    """Converts numeric predictions back to original string classes."""
    problem_type = pipeline.get("problem_type")
    target_col = pipeline.get("target_col")
    enc_meta = pipeline.get("enc_meta", {})
    
    if problem_type == "Classification" and target_col in enc_meta:
        stored_map = enc_meta[target_col].get("stored_map", {})
        inv_map = {v: k for k, v in stored_map.items()}
        return inv_map.get(pred, pred)
    return pred

def main():
    st.title("🎯 Gradient Boosting Predictor")
    st.write("Task: **Classification**")

    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Could not load model pipeline file: {e}")
        return

    model = pipeline.get('model')
    ui_cols = pipeline.get('required_original_cols', [])
    ui_metadata = pipeline.get('ui_metadata', {})
    target_col = pipeline.get('target_col', 'target')
    target_transform = pipeline.get('target_transform', 'Keep as it is')

    # --- Sidebar: Batch Prediction ---
    st.sidebar.header("📂 Batch Prediction")
    st.sidebar.caption("Upload a CSV file containing the original unencoded features.")
    batch_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if batch_file:
        st.sidebar.info("Processing...")
        df_batch = pd.read_csv(batch_file)
        
        try:
            df_processed = preprocess_data(df_batch, pipeline)
            preds = model.predict(df_processed.values.astype(np.float64))
            
            if target_transform == "Apply inverse log (np.expm1)" and "Classification" == "Regression":
                preds = np.expm1(preds)
            elif "Classification" == "Classification":
                preds = [decode_target(p, pipeline) for p in preds]
                
            df_batch[f"Predicted_{target_col}"] = preds
            st.sidebar.success("✅ Batch prediction complete!")
            st.sidebar.download_button(
                label="⬇️ Download Predictions", 
                data=df_batch.to_csv(index=False), 
                file_name="predictions.csv", 
                mime="text/csv"
            )
            st.subheader("Batch Prediction Results")
            st.dataframe(df_batch.head(20), use_container_width=True)
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")
        return

    # --- Main Page: Single Prediction UI ---
    st.markdown("### ✨ Enter Feature Values")
    st.caption("Adjust the values below. Categorical values have been mapped back to their original text.")

    with st.form("predict_form"):
        n_cols = 3
        grid = [ui_cols[i:i + n_cols] for i in range(0, len(ui_cols), n_cols)]
        input_dict = {}

        for row_cols in grid:
            cols = st.columns(n_cols)
            for col_ui, feature in zip(cols, row_cols):
                with col_ui:
                    meta = ui_metadata.get(feature, {})
                    
                    if meta.get("type") == "categorical":
                        opts = meta.get("options", [""])
                        input_dict[feature] = st.selectbox(
                            label=f"{feature}",
                            options=opts,
                            index=0
                        )
                    else:
                        val = meta.get("mean", 0.0)
                        input_dict[feature] = st.number_input(
                            label=f"{feature}",
                            value=float(val),
                            format="%.4f"
                        )

        submit = st.form_submit_button("🔮 Predict", type="primary")

    if submit:
        df_input = pd.DataFrame([input_dict])
        
        try:
            df_processed = preprocess_data(df_input, pipeline)
            pred = model.predict(df_processed.values.astype(np.float64))
            
            if target_transform == "Apply inverse log (np.expm1)" and "Classification" == "Regression":
                pred = np.expm1(pred)
                
            pred_val = pred[0]
            if "Classification" == "Classification":
                pred_val = decode_target(pred_val, pipeline)

            st.markdown("---")
            if "Classification" == "Regression":
                st.success(f"### 🎯 Predicted {target_col}: **{pred_val:,.4f}**")
            else:
                st.success(f"### 🎯 Predicted Class: **{pred_val}**")
                
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(df_processed.values.astype(np.float64))[0]
                        classes = model.classes_
                        decoded_classes = [decode_target(c, pipeline) for c in classes]

                        st.markdown("**Class Probabilities:**")
                        for cls, p in zip(decoded_classes, probs):
                            st.write(f"{cls}: {p:.4f}")

                    except Exception as e:
                        st.warning(f"Could not compute probabilities: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()