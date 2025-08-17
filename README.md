# 🅿️ Parking Dynamic Analysis & Prediction

This project analyzes parking lot dynamics, performs feature engineering, trains predictive models, and visualizes results in an interactive Streamlit dashboard.
It covers occupancy prediction, queue classification, and insightful visualizations for better parking management.

## 🚀 Features
 - Data Preprocessing
     - Convert timestamps to local time (WIB – Asia/Jakarta)
     - Extract time-based features (hour, day of week, time category)
     - Encode categorical variables (vehicle type, traffic condition, etc.)
     - Handle holidays & special day flags
     - Cyclical encoding for time features

 - Exploratory Data Analysis (EDA)
     - Category distribution (bar chart & pie chart)
     - Occupancy by day & time (heatmaps)
     - Correlation analysis

 - Predictive Modeling
     - Occupancy Regression Model → evaluates with MAE, RMSE, MAPE
     - Queue Classification Model → evaluates with Accuracy, Recall, and Confusion Matrix
     - Feature importance visualization for both models

 - Insights & Business Metrics
     - Peak hours and busiest days analysis
     - Estimated turnover per hour by day & vehicle type
     - Predicted occupancy ratio per location (top 10 locations bar & pie chart)

 - Interactive Dashboard (Streamlit)
     - Tabs for EDA, correlation, prediction, and business insights
     - Dynamic plots (scatter, regression line, heatmaps, feature importance, etc.)
     - Side-by-side comparisons (occupancy vs prediction, classification vs importance)

📊 Example Visualizations

 - Distribution plots → Bar chart + Pie chart side by side
 - Heatmaps → Average occupancy by day & time category
 - Regression plots → Actual vs Predicted occupancy with reference line
 - Confusion matrix → Queue classification results
 - Feature importance → Top predictors for occupancy & queue


🛠️ Tech Stack

 - Python 3.12

 - Libraries:

   - pandas, numpy
   - matplotlib, seaborn
   - scikit-learn
   - streamlit
   - holidays, pytz

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/AmriDomas/Parking-Dynamic.git
   cd Parking-Dynamic
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run streamlit_parking.py
   ```

## 📈 Future Improvements

 - Add real-time data streaming (Kafka / Spark integration)
 - Deploy dashboard on cloud (Streamlit Cloud / Docker)
 - Enhance models with deep learning (LSTM for time series prediction)
 - Add API endpoints for integration with parking systems

👤 Author

Developed by Amri

- Data Science | Machine Learning Engineer | AI Engineer | Project Manager
- [Linkedin](https://www.linkedin.com/in/muh-amri-sidiq/)
- [Kaggle](https://www.kaggle.com/amri11)
