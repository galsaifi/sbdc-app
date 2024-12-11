import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import io


# Convert DataFrame to downloadable Excel file
@st.cache_data
def convert_df(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

def download_plot_as_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# Function to preprocess the uploaded data
def preprocess_data(file):
    humanCapital = pd.read_excel(file)
    return humanCapital

def add_spacing():
    st.write("\n")  # Add empty lines for spacing
    st.divider()

# Function to train model and generate predictions
def get_counseling_hours(data):
    # Select the predictor variables and target variable
    X_hours = data[['Total Staff', 'Population', 'Is Urban', 'HR Support', 
                    'Number of Clients', 'Has Employees Reporting']]
    y_hours = data['Total Counseling Time']

    # Define a log transformer and a preprocessing pipeline
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    preprocessor = ColumnTransformer(
        transformers=[('log', log_transformer, ['Population', 'Total Staff'])],
        remainder='passthrough'
    )

    # Build the pipeline with a Random Forest model
    pipeline_hours = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', 
                                        min_samples_split=2, min_samples_leaf=1, random_state=42))
    ])

    # Train the model pipeline
    pipeline_hours.fit(X_hours, y_hours)

    # Predict counseling hours
    predicted_hours = pipeline_hours.predict(X_hours)
    data['Expected Counseling Hours'] = np.round(predicted_hours, 2)

    # Calculate metrics for additional insights
    data['Hours Per Consultant'] = np.round(data['Expected Counseling Hours'] / data['Total Staff'], 2)
    data['Hours Per Client'] = np.round(data['Expected Counseling Hours'] / data['Number of Clients'], 2)

    return data


# Sensitivity analysis and calculate Ideal Staff
def perform_sensitivity_analysis(data):
    data = data.copy()
    
    data = data[(data['Total Staff'] > 0) & 
                                      (data['Number of Clients'] > 0) & 
                                      (data['Total Counseling Time'] > 0)]

    # Define X and y
    X_staff = data[['Total Counseling Time', 'Population', 'Is Urban', 
                               'HR Support', 'Number of Clients', 'Has Employees Reporting']]
    y_staff = data['Total Staff']

    # Define pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[('log', FunctionTransformer(np.log1p, validate=True), 
                           ['Population', 'Total Counseling Time'])], remainder='passthrough')),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', 
                                        min_samples_split=2, min_samples_leaf=1, random_state=42))
    ])
    pipeline.fit(X_staff, y_staff)

    # Sensitivity analysis to calculate ideal staff
    adjustments = np.linspace(-10, 10, 10) 
    ideal_staff_per_center = []
    
    for _, row in data.iterrows():
        adjusted_predictions = []
        for adj in adjustments:
            adjusted_features = row.copy()
            adjusted_features['Total Counseling Time'] *= (1 + adj / 100)
            adjusted_features['Population'] *= (1 + adj / 100)
            adjusted_features['Number of Clients'] *= (1 + adj / 100)

            adjusted_data = pd.DataFrame([adjusted_features], columns=X_staff.columns)
            adjusted_prediction = pipeline.predict(adjusted_data)
            adjusted_predictions.append(adjusted_prediction[0])

        ideal_staff = np.round(np.mean(adjusted_predictions), 1)
        ideal_staff_per_center.append(ideal_staff)

    data['Ideal Staff'] = ideal_staff_per_center
    return data

# Streamlit app layout
st.title("SBDC Human Capital")

st.sidebar.header("Upload Datasets")
uploaded_file = st.sidebar.file_uploader("Upload Human Capital Data", type=["xlsx", "csv"])


if uploaded_file:
    data = preprocess_data(uploaded_file)
    st.write("### Data Preview")
    data.index = np.arange(1, len(data) + 1)
    st.dataframe(data)

    # Create tabs
    tab1, tab2 = st.tabs(["Counseling Time Analysis", "Staff Analysis"])

    # Tab 1: Prediction Results
    with tab1:
        # Generate prediction results
        results = get_counseling_hours(data)

        # Display prediction results table
        st.write("### Prediction Results")
        st.dataframe(results[['State', 'Center Number', 'Center Name', 'Total Staff', 'Number of Clients', 'Total Counseling Time', 
                              'Expected Counseling Hours', 'Hours Per Consultant', 
                              'Hours Per Client']])
        
        # Download link for prediction results
        st.download_button(
            label="Download Results",
            data=convert_df(results),
            file_name="Counseling_Hours_Results.xlsx",
            mime="application/vnd.ms-excel"
        )

        add_spacing()

        # Plot: Hours Per Consultant by Center
        state_filter = st.selectbox("Select State for Analysis", options=results['State'].unique(), index=0)
        filtered_data = results[results['State'] == state_filter].copy()
        
        if state_filter == "Nebraska":
            filtered_data['Center Label'] = filtered_data['Center Name'].astype(str).fillna("Unknown")
        else:
            filtered_data['Center Label'] = "Center " + filtered_data['Center Number'].astype(str)
        
        st.write(f"### Hours Per Consultant for Centers in {state_filter}")
        fig, ax = plt.subplots(figsize=(14, 10))
        bars = ax.bar(filtered_data['Center Label'], filtered_data['Hours Per Consultant'], color='royalblue')

        ax.set_title(f'Hours Per Consultant for {state_filter} Centers', fontsize=16, pad=15)
        ax.set_xlabel('Center', fontsize=14, labelpad=20)
        ax.set_ylabel('Hours Per Consultant', fontsize=14, labelpad=20)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{np.round(height, 2)}', ha='center', fontsize=12)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        png_buffer = download_plot_as_png(fig)
        st.download_button(
            label="Download Plot",
            data=png_buffer,
            file_name=f"{state_filter}_Hours_Per_Consultant.png",
            mime="image/png"
        )

        add_spacing()

        # Average Hours Per Consultant and per Client by State
        state_comparisons = results.groupby("State")[['Hours Per Consultant', 'Hours Per Client']].mean()
        
        st.write("### Average Hours Per Consultant and per Client by State")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].bar(state_comparisons.index, state_comparisons['Hours Per Consultant'], color='#0061a0')
        axes[0].set_title('Average Hours Per Consultant by State', fontsize=16, pad=15)
        axes[0].set_xlabel('State', fontsize=14, labelpad=20)
        axes[0].set_ylabel('Hours Per Consultant', fontsize=14, labelpad=20)
        for i, v in enumerate(state_comparisons['Hours Per Consultant']):
            axes[0].text(i, v + 5, str(np.round(v, 2)), ha='center', fontsize=12)
        
        axes[1].bar(state_comparisons.index, state_comparisons['Hours Per Client'], color='purple')
        axes[1].set_title('Average Hours Per Client by State', fontsize=16, pad=15)
        axes[1].set_xlabel('State', fontsize=14, labelpad=20)
        axes[1].set_ylabel('Hours Per Client', fontsize=14, labelpad=20)
        for i, v in enumerate(state_comparisons['Hours Per Client']):
            axes[1].text(i, v + 0.03, str(np.round(v, 2)), ha='center', fontsize=12)

        plt.setp(axes, xticks=range(len(state_comparisons.index)), xticklabels=state_comparisons.index)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        png_buffer = download_plot_as_png(fig)
        st.download_button(
            label="Download Plot",
            data=png_buffer,
            file_name="Average_Hours_Per_Consultant_Client.png",
            mime="image/png"
        )

    # Tab 2: Ideal Staff Number Analysis
    with tab2:
        st.write("### Ideal Staff Number Analysis")
        ideal_staff_data = perform_sensitivity_analysis(data)
        
        # Display ideal staff analysis results table
        st.dataframe(ideal_staff_data[['State', 'Center Number', 'Center Name', 'Total Staff', 'Ideal Staff']])
        
        # Download link for Ideal Staff analysis results
        st.download_button(
            label="Download Results",
            data=convert_df(ideal_staff_data),
            file_name="Ideal_Staff_Analysis_Results.xlsx",
            mime="application/vnd.ms-excel"
        )