import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import optuna
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import openpyxl
from datetime import datetime
import os
import joblib
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class DepressionPredictor:
    def __init__(self):
        self.numerical_features = ['AGE', 'WEIGHT', 'HEIGHT', 'ENGAGEMENT', 'EXCITEMENT', 
                                 'LTE', 'STRESS', 'RELAXATION', 'INTEREST', 'FOCUS', 'SLEEP']
        self.categorical_features = ['GENDER', 'LIFESTYLE', 'BMI', 'WORK TIME', 
                                   'ENERGY LEVEL', 'OBSERVED PRODUCTIVITY']
        self.target = 'STATE OF HEALTH'
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        
    def load_data(self, file):
        file_extension = file.name.split('.')[-1].lower()
        try:
            if file_extension == 'csv':
                data = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(file, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            required_columns = self.numerical_features + self.categorical_features + [self.target]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            return data
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def preprocess_data(self, data):
        processed_data = data.copy()
        categorical_mappings = {
            'BMI': ['underweight', 'normal', 'overweight', 'obese'],
            'WORK TIME': ['low', 'normal', 'high'],
            'ENERGY LEVEL': ['low', 'medium', 'high'],
            'OBSERVED PRODUCTIVITY': ['low', 'medium', 'high']
        }
        
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if col in categorical_mappings:
                    self.label_encoders[col].fit(categorical_mappings[col])
                else:
                    self.label_encoders[col].fit(processed_data[col].astype(str))
            
            try:
                processed_data[col] = self.label_encoders[col].transform(processed_data[col].astype(str))
            except ValueError:
                default_value = self.label_encoders[col].transform([categorical_mappings[col][0] if col in categorical_mappings else processed_data[col].mode()[0]])[0]
                processed_data[col] = default_value
        
        if self.target in processed_data.columns:
            processed_data[self.target] = processed_data[self.target].astype(int)
        return processed_data
    
    def create_pipeline(self):
        numeric_transformer = StandardScaler()
        self.preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, self.numerical_features + self.categorical_features)]
        )
        
        xgb_classifier = XGBClassifier(
            objective='multi:softmax',
            num_class=5,
            random_state=42,
            use_label_encoder=False,
            tree_method='hist'
        )
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb_classifier)
        ])
    
    def objective(self, trial, X, y):
        param = {
            'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'classifier__max_depth': trial.suggest_int('max_depth', 3, 10),
            'classifier__learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'classifier__subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'classifier__colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'classifier__min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
        }
        self.model.set_params(**param)
        return cross_val_score(self.model, X, y, cv=5, scoring='accuracy').mean()
    
    def train(self, X, y):
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            default_params = {
                'classifier__n_estimators': 200,
                'classifier__max_depth': 6,
                'classifier__learning_rate': 0.01,
                'classifier__subsample': 0.8,
                'classifier__colsample_bytree': 0.8,
                'classifier__min_child_weight': 3
            }
            
            try:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self.objective(trial, X_resampled, y_resampled), 
                             n_trials=10, n_jobs=-1)
                best_params = study.best_params
                params = {f'classifier__{k}': v for k, v in best_params.items()}
            except (ValueError, optuna.exceptions.TrialPruned):
                params = default_params
                st.warning("Using default parameters as optimization failed")
            
            self.model.set_params(**params)
            self.model.fit(X_resampled, y_resampled)
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        if not hasattr(self.model, 'named_steps') or not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            raise RuntimeError("Model must be fitted before evaluation")
        try:
            y_pred = self.model.predict(X_test)
            return classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, X):
        if not hasattr(self.model, 'named_steps') or not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_feature_importance(self):
        if not hasattr(self.model, 'named_steps') or not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            raise RuntimeError("Model must be fitted before getting feature importance")
        return pd.DataFrame({
            'feature': self.numerical_features + self.categorical_features,
            'importance': self.model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)

def analyze_text_sentiment(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the following text for depression indicators. Return a score between 0 and 1, where 1 indicates severe depression signs and 0 indicates no depression signs."},
                {"role": "user", "content": text}
            ]
        )
        return float(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return 0.5

def create_spider_plot(metrics):
    categories = list(metrics.keys())
    values = list(metrics.values())
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#1f77b4')
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400
    )
    return fig

def calculate_phq9_severity(score):
    if score < 5:
        return "Minimal Depression"
    elif score < 10:
        return "Mild Depression"
    elif score < 15:
        return "Moderate Depression"
    elif score < 20:
        return "Moderately Severe Depression"
    else:
        return "Severe Depression"

def save_model(model, path="model"):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "depression_model.joblib")
    try:
        # Save only the necessary components
        model_components = {
            'model': model.model,
            'label_encoders': model.label_encoders,
            'preprocessor': model.preprocessor
        }
        joblib.dump(model_components, model_path)
        st.success(f"Model saved successfully to {model_path}")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def load_saved_model(path="model"):
    model_path = os.path.join(path, "depression_model.joblib")
    if os.path.exists(model_path):
        try:
            components = joblib.load(model_path)
            predictor = DepressionPredictor()
            predictor.model = components['model']
            predictor.label_encoders = components['label_encoders']
            predictor.preprocessor = components['preprocessor']
            return predictor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    return None

@st.cache_resource
def load_model():
    """
    Load or create a depression predictor model.
    Returns initialized DepressionPredictor instance.
    """
    model_path = os.path.join("model", "depression_model.joblib")
    predictor = DepressionPredictor()
    
    if os.path.exists(model_path):
        try:
            components = joblib.load(model_path)
            if isinstance(components, dict):
                predictor.model = components.get('model')
                predictor.label_encoders = components.get('label_encoders', {})
                predictor.preprocessor = components.get('preprocessor')
                if all([predictor.model, predictor.label_encoders, predictor.preprocessor]):
                    return predictor
            elif isinstance(components, Pipeline):
                predictor.model = components
                return predictor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    return predictor


def create_confusion_matrix_plot(conf_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted ' + str(i) for i in range(conf_matrix.shape[1])],
        y=['Actual ' + str(i) for i in range(conf_matrix.shape[0])],
        colorscale='Blues'
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    return fig

def main():
    st.set_page_config(page_title="Depression Prediction System", layout="wide")
    
    st.markdown("""
        <h1 style='text-align: center; background-color: rgba(0, 123, 255, 0.1); 
        padding: 20px; border-radius: 10px;'>
        🧠 Smart Analysis of Neural Variations for Emotion Evaluation
        </h1>
        """, unsafe_allow_html=True)
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Model Training", "Prediction (Real-time)", "Model Insights"])
    
    predictor = load_model()
    
    if page == "Model Training":
        st.header("Upload dataset and Train Model")
        
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            data = predictor.load_data(uploaded_file)
            if data is not None:
                st.success("Data uploaded successfully!")
                
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                if st.button("Train Model"):
                    with st.spinner("Training model... This might take a few minutes."):
                        processed_data = predictor.preprocess_data(data)
                        X = processed_data[predictor.numerical_features + predictor.categorical_features]
                        y = processed_data[predictor.target]
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        predictor.create_pipeline()
                        predictor.train(X_train, y_train)
                        
                        st.subheader("Model Performance")
                        report, conf_matrix = predictor.evaluate(X_test, y_test)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text("Classification Report")
                            st.text(report)
                        
                        with col2:
                            st.plotly_chart(create_confusion_matrix_plot(conf_matrix))
                        
                        st.session_state['model'] = predictor
                        save_model(predictor)
                        st.success("Model trained successfully!")
    
    elif page == "Prediction (Real-time)":
        st.header("Depression Risk Assessment")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model first!")
            return
            
        tab1, tab2, tab3 = st.tabs(["Personal & EEG Metrics", "PHQ-9 Assessment", "Additional Context"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Personal Information")
                age = st.number_input("Age", 18, 100)
                gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                weight = st.number_input("Weight (kg)", 40, 200)
                height = st.number_input("Height (cm)", 140, 220)
                
            with col2:
                st.subheader("Lifestyle Metrics")
                lifestyle = st.selectbox("Sleep Pattern", [0, 1], 
                                       format_func=lambda x: "Late Sleeper" if x == 0 else "Early Riser")
                sleep = st.slider("Sleep Duration (hours)", 4, 12)
                current_time = st.time_input("Current Work Time")
                work_time = "normal" if 9 <= current_time.hour <= 17 else "high" if current_time.hour > 17 else "low"
                energy_level = st.selectbox("Energy Level", ["low", "medium", "high"])
                
            with col3:
                st.subheader("EEG Metrics")
                engagement = st.slider("Engagement", 0.0, 1.0)
                excitement = st.slider("Excitement", 0.0, 1.0)
                lte = st.slider("LTE", 0.0, 1.0)
                stress = st.slider("Stress", 0.0, 1.0)
                relaxation = st.slider("Relaxation", 0.0, 1.0)
                interest = st.slider("Interest", 0.0, 1.0)
                focus = st.slider("Focus", 0.0, 1.0)
            
            st.subheader("Workplace Assessment")
            productivity = st.selectbox("Productivity at Workplace", ["low", "medium", "high"])

        with tab2:
            st.subheader("PHQ-9 Assessment")
            st.markdown("Over the last 2 weeks, how often have you been bothered by any of the following problems?")
            
            phq9_questions = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling/staying asleep, sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself - or that you are a failure",
                "Trouble concentrating on things",
                "Moving or speaking so slowly that other people could have noticed",
                "Thoughts that you would be better off dead or of hurting yourself"
            ]
            
            if 'phq9_scores' not in st.session_state:
                st.session_state.phq9_scores = [0] * len(phq9_questions)
            
            for i, question in enumerate(phq9_questions):
                score = st.select_slider(
                    f"{i + 1}. {question}",
                    options=[0, 1, 2, 3],
                    value=st.session_state.phq9_scores[i],
                    format_func=lambda x: {0: "Not at all", 1: "Several days", 
                                         2: "More than half the days", 3: "Nearly every day"}[x],
                    key=f"phq9_{i}"
                )
                st.session_state.phq9_scores[i] = score
        
        with tab3:
            st.subheader("Additional Context (Required)")
            additional_thoughts = st.text_area(
                "Share your thoughts, feelings, and current mental state:",
                height=150
            )
            if not additional_thoughts:
                st.warning("⚠️ Please provide additional context for accurate assessment")
        
        if st.button("Generate Assessment") and additional_thoughts:
            bmi_value = weight / ((height/100) ** 2)
            bmi = "underweight" if bmi_value < 18.5 else "normal" if bmi_value < 25 else "overweight" if bmi_value < 30 else "obese"
            
            input_data = pd.DataFrame({
                'AGE': [age], 'WEIGHT': [weight], 'HEIGHT': [height], 'GENDER': [gender],
                'LIFESTYLE': [lifestyle], 'ENGAGEMENT': [engagement], 'EXCITEMENT': [excitement],
                'LTE': [lte], 'STRESS': [stress], 'RELAXATION': [relaxation],
                'INTEREST': [interest], 'FOCUS': [focus], 'SLEEP': [sleep],
                'BMI': [bmi], 'WORK TIME': [work_time], 'ENERGY LEVEL': [energy_level],
                'OBSERVED PRODUCTIVITY': [productivity]
            })
            
            processed_input = predictor.preprocess_data(input_data)
            prediction = predictor.predict(processed_input)
            prediction_value = int(prediction[0])
            
            sentiment_score = analyze_text_sentiment(additional_thoughts)
            phq9_total = sum(st.session_state.phq9_scores)
            phq9_severity = calculate_phq9_severity(phq9_total)
            
            combined_risk = (prediction_value/5 + phq9_total/27 + sentiment_score) / 3
            risk_percentage = int(combined_risk * 100)
            
            st.header("Prediction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Clinical Summary")
                clinical_status = "Depressed" if combined_risk > 0.5 else "Not Depressed"
                st.metric("Clinical Status", clinical_status)
                severity_color = "red" if combined_risk > 0.5 else "green"
                st.markdown(f"<p style='color: {severity_color}; font-size: 18px;'>Risk Score: {risk_percentage}%</p>", 
                          unsafe_allow_html=True)
                
                st.subheader("Risk Level Analysis")
                risk_text = "Low" if risk_percentage < 30 else "Moderate" if risk_percentage < 60 else "High"
                st.progress(risk_percentage/100)
                st.write(f"Current Risk Level: **{risk_text}**")
                
            with col2:
                metrics = {
                    'Sleep Quality': sleep/12,
                    'Stress Management': 1-stress,
                    'Engagement': engagement,
                    'Focus': focus,
                    'Energy': {'low': 0.3, 'medium': 0.6, 'high': 0.9}[energy_level]
                }
                st.subheader("Wellness Metrics")
                st.plotly_chart(create_spider_plot(metrics))
            
            st.subheader("Mood Trend Analysis")
            col1, col2 = st.columns(2)
            with col1:
                mood_scores = st.session_state.phq9_scores
                fig = px.line(x=list(range(1, 10)), y=mood_scores,
                            labels={'x': 'PHQ-9 Question Number', 'y': 'Score'},
                            title='PHQ-9 Response Pattern')
                st.plotly_chart(fig)
            
            with col2:
                st.markdown(f"""
                ### PHQ-9 Assessment Summary
                - Total Score: **{phq9_total}/27**
                - Severity Level: **{phq9_severity}**
                - Primary Concerns: {', '.join([phq9_questions[i] for i, score in enumerate(mood_scores) if score >= 2])}
                """)
            
            st.subheader("Recommendations & Lifestyle Suggestions")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Clinical Recommendations")
                recommendations = []
                
                if risk_percentage >= 60:
                    recommendations.append("🚨 Immediate professional consultation recommended")
                if phq9_total >= 15:
                    recommendations.append("👨‍⚕️ Schedule an appointment with a mental health specialist")
                if stress > 0.7:
                    recommendations.append("🧘‍♀️ Consider stress management therapy or counseling")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            with col2:
                st.markdown("### Lifestyle Suggestions")
                lifestyle_suggestions = []
                
                if sleep < 7:
                    lifestyle_suggestions.append("🛏️ Improve sleep hygiene - aim for 7-9 hours")
                if bmi != "normal":
                    lifestyle_suggestions.append("🥗 Consider consulting a nutritionist")
                if engagement < 0.5:
                    lifestyle_suggestions.append("🎯 Engage in activities you previously enjoyed")
                if relaxation < 0.5:
                    lifestyle_suggestions.append("🧘 Practice daily mindfulness exercises")
                
                for suggestion in lifestyle_suggestions:
                    st.markdown(f"- {suggestion}")
            
            save_model(st.session_state['model'])
    
    else:  # Model Insights
        st.header("Model Insights")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model first!")
            return
        
        importance_df = st.session_state['model'].get_feature_importance()
        
        st.subheader("Feature Importance Analysis")
        fig = px.bar(importance_df, x='feature', y='importance',
                    title='Feature Impact on Depression Prediction')
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Importance Score",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig)
        
        st.subheader("Feature Rankings")
        st.dataframe(importance_df)

if __name__ == "__main__":
    main()