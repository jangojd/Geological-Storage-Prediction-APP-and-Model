import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Geological Storage ML Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)


# Title
st.title("🌍 Geological Storage ML Predictor")
st.markdown("**Predict CO2 Storage Capacity using Machine Learning**")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        # Try different possible model names
        if os.path.exists('model.pkl'):
            model = joblib.load('model.pkl')
        elif os.path.exists('geological_storage_linear_regression_model.pkl'):
            model = joblib.load('geological_storage_model.pkl')
        elif os.path.exists('rf_model.pkl'):
            model = joblib.load('rf_model.pkl')
        else:
            st.error("❌ Model file not found. Please ensure your model file is uploaded.")
            return None
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

# Load scaler if exists
@st.cache_resource
def load_scaler():
    try:
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            st.sidebar.info("✅ Scaler loaded")
            return scaler
        return None
    except:
        return None

# Initialize
model = load_model()
scaler = load_scaler()

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    if model is not None:
        st.success("Model Status: ✅ Ready")
        
        # Try to get model type
        try:
            model_type = type(model).__name__
            st.info(f"Model Type: {model_type}")
        except:
            pass
    else:
        st.error("Model Status: ❌ Not Loaded")
    
    st.markdown("---")
    
    st.header("📖 About")
    st.markdown("""
    This app predicts CO2 geological storage capacity based on:
    
    - **Porosity** (0.1 - 0.4)
    - **Permeability** (50 - 250 mD)
    - **Depth** (500 - 2500 m)
    - **Pressure** (5 - 25 MPa)
    - **Temperature** (30 - 100 °C)
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("- 📓 [View Notebook](https://github.com/jangojd/Geological-Storage-Prediction-APP-and-Model/blob/main/Geological_Storage_prediction.ipynb)")
    st.markdown("- 💻 [GitHub Repo](https://github.com/YOUR_USERNAME/geological-storage-ml)")
    st.markdown("**👨‍💻 Author:** Jawad Ali")

# Main content
if model is not None:
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions", "ℹ️ Help"])
    
    with tab1:
        st.header("Make a Single Prediction")
        st.markdown("Enter the geological parameters below:")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rock Properties")
            porosity = st.slider(
                "Porosity", 
                min_value=0.1, 
                max_value=0.4, 
                value=0.25, 
                step=0.01,
                help="Fraction of pore space in rock (0.1 - 0.4)"
            )
            
            permeability = st.slider(
                "Permeability (mD)", 
                min_value=50.0, 
                max_value=250.0, 
                value=150.0, 
                step=5.0,
                help="Rock permeability in millidarcies"
            )
            
            depth = st.slider(
                "Depth (m)", 
                min_value=500.0, 
                max_value=2500.0, 
                value=1500.0, 
                step=50.0,
                help="Depth of the reservoir in meters"
            )
        
        with col2:
            st.subheader("Reservoir Conditions")
            pressure = st.slider(
                "Pressure (MPa)", 
                min_value=5.0, 
                max_value=25.0, 
                value=15.0, 
                step=0.5,
                help="Reservoir pressure in megapascals"
            )
            
            temperature = st.slider(
                "Temperature (°C)", 
                min_value=30.0, 
                max_value=100.0, 
                value=60.0, 
                step=1.0,
                help="Reservoir temperature in Celsius"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Storage Capacity", use_container_width=True):
            with st.spinner("Calculating prediction..."):
                try:
                    # Prepare input data
                    input_data = np.array([[porosity, permeability, depth, pressure, temperature]])
                    
                    # Apply scaling if scaler exists
                    if scaler is not None:
                        input_data = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    
                    # Display result
                    st.success("### ✅ Prediction Complete!")
                    
                    # Show prediction in a nice format
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Predicted Storage Capacity",
                            value=f"{prediction:.2f} tons",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence",
                            value="High" if 100 < prediction < 500 else "Medium"
                        )
                    
                    with col3:
                        st.metric(
                            label="Input Features",
                            value="5"
                        )
                    
                    # Show input summary
                    st.markdown("---")
                    st.subheader("📋 Input Summary")
                    
                    summary_df = pd.DataFrame({
                        'Parameter': ['Porosity', 'Permeability', 'Depth', 'Pressure', 'Temperature'],
                        'Value': [porosity, permeability, depth, pressure, temperature],
                        'Unit': ['', 'mD', 'm', 'MPa', '°C']
                    })
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("💡 Interpretation")
                    
                    if prediction > 30000:
                        st.success("✅ **High Storage Capacity**: This reservoir shows excellent potential for CO2 storage.")
                    elif prediction > 15000:
                        st.info("ℹ️ **Moderate Storage Capacity**: This reservoir has good potential for CO2 storage.")
                    else:
                        st.warning("⚠️ **Low Storage Capacity**: This reservoir may have limited CO2 storage potential.")
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.info("💡 Tip: Make sure your model was trained with the same feature order.")
    
    with tab2:
        st.header("Batch Predictions from CSV")
        st.markdown("Upload a CSV file with multiple geological samples for batch predictions.")
        
        # Show expected format
        with st.expander("📋 Click to see expected CSV format"):
            st.markdown("""
            Your CSV should have these columns (in this order):
            1. `porosity` (0.1 - 0.4)
            2. `permeability` (50 - 250)
            3. `depth` (500 - 2500)
            4. `pressure` (5 - 25)
            5. `temperature` (30 - 100)
            
            **Column names can be:**
            - `porosity, permeability, depth, pressure, temperature`
            - OR `porosity, permeability_md, depth_m, pressure_mpa, temperature_c`
            """)
            
            # Sample data
            sample_df = pd.DataFrame({
                'porosity': [0.25, 0.30, 0.20, 0.35],
                'permeability': [150, 180, 120, 200],
                'depth': [1500, 1800, 1200, 2000],
                'pressure': [15, 18, 12, 20],
                'temperature': [60, 70, 50, 80]
            })
            
            st.dataframe(sample_df, use_container_width=True)
            
            # Download sample
            csv = sample_df.to_csv(index=False)
            st.download_button(
                "📥 Download Sample CSV Template",
                csv,
                "sample_input.csv",
                "text/csv",
                use_container_width=True
            )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with geological parameters"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully: **{len(df)} rows**")
                
                # Show preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Predict button
                if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Extract features (handle different column name formats)
                            if 'porosity' in df.columns:
                                X = df[['porosity', 'permeability', 'depth', 'pressure', 'temperature']].values
                            else:
                                # Try alternative column names
                                X = df.iloc[:, :5].values
                            
                            # Apply scaling if needed
                            if scaler is not None:
                                X = scaler.transform(X)
                            
                            # Make predictions
                            predictions = model.predict(X)
                            
                            # Add predictions to dataframe
                            df['predicted_capacity_tons'] = predictions
                            
                            st.success("✅ Predictions completed successfully!")
                            
                            # Show results
                            st.subheader("📈 Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Statistics
                            st.subheader("📊 Summary Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Total Samples", len(predictions))
                            col2.metric("Mean Capacity", f"{predictions.mean():.2f} tons")
                            col3.metric("Max Capacity", f"{predictions.max():.2f} tons")
                            col4.metric("Min Capacity", f"{predictions.min():.2f} tons")
                            
                            # Download results
                            st.markdown("---")
                            csv_result = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results CSV",
                                csv_result,
                                "prediction_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.info("💡 Make sure your CSV has the correct columns in the right order.")
                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    with tab3:
        st.header("ℹ️ How to Use This App")
        
        st.markdown("""
        ### 🎯 Single Prediction
        1. Go to the **Single Prediction** tab
        2. Adjust the sliders for each geological parameter
        3. Click the **Predict Storage Capacity** button
        4. View your prediction and interpretation
        
        ### 📊 Batch Predictions
        1. Go to the **Batch Predictions** tab
        2. Download the sample CSV template
        3. Fill in your data (keep the same format)
        4. Upload your CSV file
        5. Click **Run Batch Predictions**
        6. Download the results
        
        ### 📝 Parameter Ranges
        - **Porosity**: 0.1 - 0.4 (fraction of pore space)
        - **Permeability**: 50 - 250 mD (ability of fluid flow)
        - **Depth**: 500 - 2500 m (reservoir depth)
        - **Pressure**: 5 - 25 MPa (reservoir pressure)
        - **Temperature**: 30 - 100 °C (reservoir temperature)
        
        ### ❓ FAQ
        
        **Q: What model is being used?**  
        A: This app uses a pre-trained machine learning model (check sidebar for model type).
        
        **Q: How accurate are the predictions?**  
        A: Accuracy depends on how well the training data represents your geological formation.
        
        **Q: Can I use my own data?**  
        A: Yes! Use the Batch Predictions feature to upload a CSV with your data.
        
        **Q: What if my prediction seems wrong?**  
        A: Ensure your input values are within the valid ranges and represent realistic geological conditions.
        """)
        
        st.markdown("---")
        st.info("💡 **Tip**: For best results, ensure your input parameters are within the specified ranges and represent realistic geological conditions.")

else:
    st.error("⚠️ **Model Not Found**")
    st.markdown("""
    ### Please ensure:
    1. Your model file (`model.pkl`) is in the same directory as `app.py`
    2. The model was saved using `joblib` or `pickle`
    3. The file permissions allow reading
    
    ### To save your model:
    ```python
    import joblib
    joblib.dump(your_model, 'model.pkl')
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🌍 Geological Storage ML Predictor | Built with Streamlit & Scikit-learn</p>
        <p>Predicting CO2 Storage Capacity for a Sustainable Future</p>
    </div>
""", unsafe_allow_html=True)
