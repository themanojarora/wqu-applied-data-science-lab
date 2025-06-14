# Applied Data Science Lab @ WorldQuant University (2025)

## 📜 Overview
This repository documents my successful completion of the **Applied Data Science Lab**, an intensive 16-week online course by [WorldQuant University](https://wqu.org). The program is structured around 8 end-to-end, real-world projects covering the full lifecycle of data science workflows—from data ingestion and cleaning to machine learning, model evaluation, and deployment.

> ⚠️ **Disclaimer**: Due to WQU's licensing and academic integrity policy, no code or datasets from the course are shared in this repository.

---

## 📌 Program Map
![Program Map](./8a32c356-709d-44df-8563-d32d508ad780.png)

---

## 🔍 Summary of Projects
Below is a detailed breakdown of the 8 projects and the skills/tools applied in each:

### 1. Housing in Mexico
**Objective**: To analyze and visualize regional variations in housing prices across Brazil using structured property data and geospatial mapping.

**Technologies & Tools**:
- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly.express`
- **Visualization**: `scatter_mapbox`, histograms, boxplots, bar charts
- **Data Engineering**: `read_csv`, column parsing, string cleaning, null handling, currency conversion
- **Analytical Methods**: Descriptive stats, correlation matrices, grouped aggregation, geospatial visual analytics

**Key Contributions**:
- Merged two distinct CSV datasets from different sources and reconciled column naming, currency formats, and geolocation markers
- Engineered new features such as state names from address strings and latitude-longitude fields for spatial plotting
- Normalized prices by converting BRL to USD for comparative regional analysis
- Identified key correlations between property price, area, and location—both across states and by geographic coordinates
- Created visual dashboards highlighting disparities between densely populated and high-cost regions like the South and Central-West
- Structured a robust EDA pipeline that used `.groupby()`, `.describe()`, and advanced filtering to explore property trends

**Conceptual Learning**:
- Dealing with real-world inconsistencies in datasets (currency, units, formats)
- Deriving business insight from region-tagged numeric data
- Best practices for reproducible EDA

---

### 2. Housing in Buenos Aires
**Objective**: Build a predictive model to estimate apartment prices in Mexico City using real-world listings, while refining data preprocessing, pipeline modeling, and evaluation workflows.

**Technologies & Tools**:
- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Modeling**: `Ridge`, `LinearRegression`, `SimpleImputer`, `OneHotEncoder`, `Pipeline`
- **Visualization**: heatmaps, histograms, bar charts, scatter plots

**Key Contributions**:
- Created a reusable `wrangle()` function to preprocess raw CSV files:
  - Removed outliers and non-Mexico City data
  - Extracted boroughs from address strings
  - Converted local currency to USD
  - Cleaned and normalized categorical variables
- Merged multiple datasets into a unified DataFrame of properties
- Conducted correlation analysis with heatmaps and scatter plots
- Split data into train/test and constructed baseline and regularized regression models
- Designed a modular modeling pipeline using `Pipeline()` to simplify transformation and prediction
- Evaluated model performance on a holdout set and interpreted feature importances
- Created a final ranked visualization of which borough features contributed most to predicted apartment prices

**Conceptual Learning**:
- Best practices for robust data wrangling with real estate datasets
- Feature engineering for location-based data (borough, lat/lon)
- Importance of pipeline abstraction in production-ready machine learning
- Metrics interpretation (MAE) and coefficient-based model explainability

### 3. Air Quality Forecast in Nairobi
**Objective**: To forecast PM2.5 air pollution levels using time-series data collected from air quality sensors in Dar es Salaam and apply autoregressive modeling with walk-forward validation.

**Technologies & Tools**:
- **Database**: MongoDB via `pymongo`
- **Libraries**: `pandas`, `matplotlib`, `statsmodels`, `sklearn`
- **Visualization**: line plots, autocorrelation (ACF), partial autocorrelation (PACF), histogram plots
- **Modeling**: AR models, walk-forward validation, residual analysis

**Key Contributions**:
- Connected to a local MongoDB server to extract raw air quality data from multiple sensor sites
- Preprocessed time-series data with timezone localization, interpolation of missing values, and datetime indexing
- Generated time series plots and 7-day rolling averages to understand trends in PM2.5 concentration
- Built and analyzed ACF and PACF plots to determine the stationarity and lag order of the data
- Trained autoregressive (AR) models with different hyperparameter settings and evaluated performance using Mean Absolute Error (MAE)
- Used `statsmodels` ARIMA/AR framework to build, fit, and compare models
- Conducted walk-forward validation over a test set and visualized prediction versus actual values
- Interpreted model residuals, plotted their histogram, and examined their autocorrelation for final model diagnostics

**Conceptual Learning**:
- Applied time series modeling workflows: stationarity check → lag analysis → residual diagnostics → model evaluation
- Gained familiarity with modeling under memory constraints and rolling validation strategies
- Strengthened confidence in using MongoDB in data science pipelines
- Demonstrated proficiency in model interpretation beyond metrics—through visualization and statistical diagnostics`

### 4. Earthquake Damage in Nepal
**Objective**: Predict structural damage outcomes for buildings in Nepal using post-earthquake assessment data and supervised classification models.

**Technologies & Tools**:
- **Database**: SQLite via `sqlite3` and `%sql` magic
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Modeling**: `LogisticRegression`, `DecisionTreeClassifier`, `make_pipeline`, `OrdinalEncoder`, `SimpleImputer`
- **Visualization**: bar plots, box plots, feature importance charts

**Key Contributions**:
- Connected to structured SQLite datasets and loaded relevant tables into pandas DataFrames
- Calculated and explored distributions of building types, districts, and damage grades for initial EDA
- Wrangled and cleaned data using string parsing and conditional logic; engineered new features from nested variables
- Created visual comparisons of building attributes across damage severity levels
- Split dataset into training and validation sets using stratification and reproducible seeds
- Built classification pipelines using logistic regression and decision trees, embedding preprocessing steps into the model pipeline
- Evaluated models using accuracy scores, confusion matrices, and classification reports
- Tuned hyperparameters (e.g., `max_depth`) for decision trees and interpreted model outputs using odds ratios and Gini importance
- Conducted feature importance analysis and visualized contribution of predictors

**Conceptual Learning**:
- Differentiating model interpretability between decision trees (feature splits) and logistic regression (coefficients)
- Practical exposure to encoding, imputation, and model fitting within integrated pipelines
- Reflections on the ethical risks of biased datasets in disaster management, especially concerning socioeconomic or geographic feature correlations

### 5. Bankruptcy Prediction in Poland
**Objective**: Predict corporate bankruptcy using financial indicators while handling significant class imbalance and evaluating model performance under varying thresholds.

**Technologies & Tools**:
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Modeling**: `RandomForestClassifier`, `GradientBoostingClassifier`, `SimpleImputer`, `make_pipeline`
- **Evaluation**: Confusion matrix, classification report, accuracy, precision, recall

**Key Contributions**:
- Cleaned and joined company financial data across multiple CSVs and filtered based on year indicators
- Explored correlations among financial metrics and distribution of target classes using bar plots
- Addressed **severe class imbalance** by implementing both undersampling and oversampling techniques on the training set
- Created pipelines embedding preprocessing (imputation, scaling) with modeling components
- Performed **k-fold cross-validation** to assess model generalizability and stability
- Used `GridSearchCV` to perform hyperparameter tuning, identifying best parameters for boosting models
- Computed and visualized **threshold-sensitive confusion matrices** to study tradeoffs between sensitivity and specificity
- Interpreted model output via feature importance rankings to assess which financial ratios were most predictive

**Conceptual Learning**:
- Developed understanding of how resampling techniques influence bias-variance tradeoff
- Learned to integrate and validate ensemble models within modular ML pipelines
- Applied classification diagnostics in the context of rare-event prediction (i.e., bankruptcy forecasting)

### 6. Customer Segmentation in the US
**Objective**: Segment U.S. consumers using financial data to uncover underlying behavior patterns and visualize them via clustering and PCA.

**Technologies & Tools**:
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`
- **Clustering**: `KMeans`, `PCA`, inertia and silhouette score analysis
- **Web Deployment**: Plotly Dash for building an interactive web interface

**Key Contributions**:
- Cleaned and explored a high-dimensional SCF dataset of U.S. households (~28k rows × 350+ features)
- Performed univariate filtering to retain relevant financial and demographic indicators
- Standardized and visualized distribution of variables, including income and asset levels, using bar charts and scatter plots
- Applied **KMeans clustering** to generate groupings and evaluated cluster quality using inertia and silhouette scores
- Used **Principal Component Analysis (PCA)** to reduce dimensions and plot clusters in a visually interpretable space
- Developed a multi-layered dashboard with dropdown filters, dynamic visuals, and clustering analytics
- The Dash app followed a clean 3-tier architecture: Data Access, Business Logic, and Presentation

**Conceptual Learning**:
- Intuition for clustering in high-dimensional consumer data
- Techniques for preprocessing sparse and noisy survey data
- Experience building interactive data applications for end-user analysis

### 7. A/B Testing at WQU
**Objective**: Conduct and analyze an A/B test experiment to assess email effectiveness in boosting admissions engagement using synthetic applicant data.

**Technologies & Tools**:
- **Database**: MongoDB via `pymongo`
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`
- **Modeling**: Custom OOP classes for data modeling, chi-square testing for group differences
- **Web Deployment**: Dash (3-tiered architecture)

**Key Contributions**:
- Connected to a MongoDB collection to fetch structured applicant data for analysis
- Aggregated and visualized nationality-level summaries using `groupby` and bar charts
- Designed and implemented OOP class architecture to support extract-transform-load (ETL) pipelines and experiment tracking
- Built statistical tests (chi-square) to compare response rates between experimental groups
- Computed odds ratios to quantify effect size and evaluate practical significance
- Constructed a dashboard app with tabs for demographic insights, experimental variation, and significance testing results
- Visualized participation and conversion metrics with stacked bar charts and descriptive summaries

**Conceptual Learning**:
- Designing controlled experiments with A/B logic and significance testing
- Importance of exploratory summaries before statistical inference
- Building statistical tests into dashboard workflows for continuous monitoring and analysis

### 8. Market Volatility Forecast in India
**Objective**: Forecast short-term asset volatility using historical financial data, GARCH modeling, and serve predictions through an API built with FastAPI.

**Technologies & Tools**:
- **Data Source**: AlphaVantage API
- **Libraries**: `pandas`, `statsmodels`, `arch`, `sqlite3`, `requests`
- **Modeling**: GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- **Programming Paradigm**: Test-Driven Development (TDD), Object-Oriented Programming (OOP)
- **Deployment**: `FastAPI`, `uvicorn`, RESTful endpoints

**Key Contributions**:
- Fetched historical asset data using a robust `AlphaVantageAPI` client that handles HTTP errors and formats time series for modeling
- Used SQLite as a lightweight local database to store raw and processed data
- Designed OOP modules for data ingestion (`SQLRepository`), model training (`GarchModel`), and forecast generation
- Cleaned time series, calculated returns and volatility metrics, and tested model stationarity and autocorrelation patterns
- Fit GARCH models using the `arch` library, validated parameters using AIC/BIC and residual analysis
- Built REST endpoints for:
  - Model training (`/fit`) with given asset data
  - Volatility prediction (`/forecast`) for new time windows
- Deployed the pipeline using `FastAPI` and served real-time model responses via HTTP requests

**Conceptual Learning**:
- Practical experience applying econometric models (GARCH) in a production setting
- Understanding the role of return volatility in risk management and asset pricing
- Building an end-to-end forecasting API that integrates live data access, model persistence, and frontend-ready outputs

---

## 🏁 Key Takeaways
- End-to-end experience with **real-world data science problems** involving ingestion, transformation, modeling, and deployment
- Hands-on practice with **ETL pipelines, classification, regression, time-series, clustering, and A/B testing**
- Advanced modeling with **AR, GARCH, Decision Trees, Gradient Boosting, Logistic Regression, and Ridge Regression**
- Practical application of **model validation techniques**, including walk-forward validation, k-fold CV, confusion matrices, and grid search
- System design experience with **object-oriented programming (OOP)** and **test-driven development (TDD)**
- Built and deployed **interactive dashboards** and **REST APIs** to communicate results
- Exposure to **ethical challenges** in AI and statistical learning, particularly with imbalanced or sensitive data

---

## 🧰 Tech Stack
**Languages**: Python

**Libraries & Tools**:
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `arch`
- `plotly`, `dash`, `FastAPI`, `uvicorn`, `requests`
- `sqlite3`, `pymongo`, `ipywidgets`

**Modeling Techniques**:
- Supervised Learning: Linear & Logistic Regression, Decision Trees, Random Forests, Gradient Boosting
- Unsupervised Learning: K-Means Clustering, PCA
- Time Series: AR, GARCH
- Statistical Testing: Chi-square, Odds Ratios, Cross-tabulation

**Development & Deployment**:
- Modular Pipelines (`make_pipeline`, `Pipeline`)
- Model Validation: k-fold CV, walk-forward, ACF/PACF diagnostics
- Dash Apps, REST APIs with FastAPI

**Design Patterns**:
- MVC Architecture, Object-Oriented Programming, TDD

---

## 🎖️ Credential
- ✅ **Badge & Certificate of Completion**: [WQU Applied Data Science Lab – Credential via Credly](https://www.credly.com/badges/2bec6470-7e4b-4bc5-b0f4-5a5d48c2470c/public_url)

---

## 💬 Let's Connect
- LinkedIn: [https://www.linkedin.com/in/themanojarora/](https://www.linkedin.com/in/themanojarora/)
- Email: [manojarorawrites@gmail.com](mailto:manojarorawrites@gmail.com)

