## Smart Analysis of Neural Variations for Emotion Evaluation

### Overview

This project implements a **Depression Prediction System** using machine learning models to assess the risk of depression based on various health and lifestyle metrics. The system uses Streamlit for a user-friendly interface, XGBoost for classification, and integrates with OpenAI for text sentiment analysis.

### Features

- **Data Loading and Preprocessing**: Supports CSV and Excel file formats, with preprocessing steps to handle categorical and numerical data.
- **Model Training**: Utilizes XGBoost classifier with hyperparameter tuning via Optuna, and SMOTE for handling imbalanced datasets.
- **Prediction**: Predicts depression risk based on input features and provides a risk score.
- **Evaluation**: Generates classification reports and confusion matrices for model performance assessment.
- **Feature Importance**: Analyzes and visualizes the importance of different features in predicting depression.
- **Sentiment Analysis**: Integrates with OpenAI's API to analyze text for depression indicators.
- **Visualization**: Includes spider plots for wellness metrics, line charts for mood trends, and bar charts for feature importance.
- **Recommendations**: Provides clinical and lifestyle suggestions based on the risk assessment.

### Installation

To run this project, you need:

- Python 3.8+
- Required libraries:
  ```bash
  pip install streamlit numpy pandas scikit-learn xgboost optuna imbalanced-learn plotly openpyxl joblib openai python-dotenv
  ```

### Usage

1. **Setup Environment Variables**: Ensure you have an `OPENAI_API_KEY` set in your environment variables or in a `.env` file.
2. **Run the Application**: 
   ```bash
   streamlit run sanvee.py
   ```
3. **Interact with the System**: 
   - Upload a dataset with the required columns.
   - The system will process the data, train the model, and provide insights and predictions.

### File Structure

- `sanvee.py`: Main script containing the DepressionPredictor class and Streamlit app logic.
- `model/`: Directory to store the trained model.
- `.env`: File to store environment variables like API keys.

### Dependencies

- `streamlit`: For creating the web application interface.
- `numpy`, `pandas`: For data manipulation.
- `scikit-learn`: For preprocessing, model selection, and evaluation.
- `xgboost`: For the classification model.
- `optuna`: For hyperparameter optimization.
- `imbalanced-learn`: For handling imbalanced datasets.
- `plotly`: For interactive visualizations.
- `openpyxl`: For reading Excel files.
- `joblib`: For model persistence.
- `openai`: For text sentiment analysis.
- `python-dotenv`: For managing environment variables.
 

 
