# Airbnb NYC 2019 Price Prediction

This project analyzes and predicts Airbnb rental prices in New York City in 2019. A key feature of this project is the **implementation from scratch** of core Machine Learning algorithms (Linear Regression, Lasso Regression) using the **NumPy** library, instead of relying on high-level libraries like Scikit-learn.

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

## 🌟 Introduction

### 1. Problem Description
The project focuses on building a Machine Learning model to predict the **listing price** of Airbnb apartments/rooms in New York City in 2019.
- **Input:** Apartment features such as location (borough, neighborhood), room type (entire home, private room), minimum nights, number of reviews, etc.
- **Output:** Rental price (continuous variable).
- **Problem Type:** Supervised Learning - Regression (Linear Regression).

### 2. Motivation & Practical Application
Dynamic Pricing is a major challenge in the sharing economy:
- **For Hosts:** Helps them set competitive prices to maximize profit and occupancy rates, avoiding pricing too high (no bookings) or too low (lost revenue).
- **For Guests:** Provides a reference to evaluate if a price is reasonable, helping them find good deals or avoid being overcharged.
- **For the Platform:** Automated price suggestions help improve user experience and balance market supply and demand.

### 3. Specific Goals
The project goes beyond just calling available libraries, aiming for in-depth objectives:
- **Technical:** Implement **Linear Regression** and **Lasso Regression** algorithms from scratch using only **NumPy**. This helps in mastering the mathematical nature (Matrix Calculus, Gradient Descent, Coordinate Descent).
- **Data:** Build a complete Data Pipeline from cleaning, noise handling, to advanced Feature Engineering like Target Encoding.
- **Results:** Build a model with acceptable accuracy (R² > 0.5) and more importantly, **interpretability** - identifying which factors most strongly affect room prices in NYC.

---

## 📊 Dataset

### 1. Data Source
- **Source:** [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- **Size:** ~49,000 data rows.

### 2. Feature Description
- **Categorical:** `neighbourhood_group` (5 boroughs), `neighbourhood` (200+ areas), `room_type` (3 types).
- **Numerical:** `latitude`, `longitude`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`.
- **Target:** `price` (USD).

### 3. Data Characteristics & EDA
Below are some important charts from the data exploration process:

#### a. Price Distribution
Most listings are priced under $200/night. The chart below shows the price distribution for apartments <= $500.
![Price Distribution](https://github.com/user-attachments/assets/4f8b7910-e4ca-497e-8e84-5582e8443ca9)

#### b. Geographic Distribution
Listings are densely concentrated in Manhattan and Brooklyn. Colors indicate price levels (red/yellow are high prices).
![Geographic Map](https://github.com/user-attachments/assets/8c8a41a7-56cf-4c81-9d52-50d040a6874b)

#### c. Price by Room Type
"Entire home/apt" has the highest price and largest variance, while "Shared room" has the lowest price.
![Room Type Price](https://github.com/user-attachments/assets/42bbe49a-5a73-4f8c-a59b-2f38d1a5b526)

#### d. Market Domination
Some areas in Manhattan are strongly dominated by "Top Hosts" (those managing multiple listings), indicating high commercialization.
![Market Domination](https://github.com/user-attachments/assets/e13c2e0e-0661-4e6e-bb5e-9fb1a4ca396a)

#### e. Correlation Matrix
The heatmap shows the correlation between variables. Variables have little strong linear correlation with each other, except for `number_of_reviews` and `reviews_per_month`.
![Correlation Matrix](https://github.com/user-attachments/assets/4ee7229f-84cb-40af-9e73-83ded285fcad)

#### f. Price Density by Neighbourhood
The Violin plot shows that the price distribution density in Manhattan is wider and has a longer tail (many high-priced listings) compared to other boroughs like Queens or Bronx.
![Price Density](https://github.com/user-attachments/assets/ec487221-866c-4d91-ab99-91ed04eb5626)

#### g. Top Words in Listing Names
The most frequent keywords in room names are often related to location ("Manhattan", "Brooklyn", "Williamsburg") and room characteristics ("Private", "Room", "Cozy", "Spacious").
![Top Words](https://github.com/user-attachments/assets/253666d1-f7df-4336-99bf-0f9564857e9a)

---

## 🛠 Method

### 1. Data Preprocessing Pipeline
The process is performed sequentially to ensure clean and informative data for the model:

#### a. Data Cleaning
- **Missing Values:**
  - `reviews_per_month`: NaN is filled with 0 (assuming no reviews means 0 reviews/month).
  - `name`, `host_name`: Missing text fields are filled with placeholders or ignored as they are not used directly.
- **Outlier Removal:**
  - Remove listings with `price = 0` (data error).
  - Remove listings with extremely high prices (e.g., > $10,000) to avoid skewing the Linear model.

#### b. Feature Engineering
- **Log Transformation:**
  - The target variable `price` is right-skewed. Apply `log(1 + price)` to bring it closer to a normal distribution, helping the linear regression model perform better.
- **Geospatial Features:**
  - Calculate the distance from the apartment to NYC center (Times Square) using the **Haversine** formula based on `latitude` and `longitude`.
- **Binning:**
  - The `minimum_nights` variable is grouped into: `short_term` (<7 days), `weekly` (7-30 days), `monthly` (>30 days) to capture different rental behaviors.

#### c. Encoding
- **One-Hot Encoding:** Applied to variables with few values (`neighbourhood_group`, `room_type`).
- **Target Encoding (with Smoothing):**
  - Applied to the `neighbourhood` variable (over 200 values). Instead of creating 200 new columns (causing data sparsity), we replace the area name with the average price of that area.
  - **Smoothing:** To avoid overfitting with areas having little data, the formula is adjusted:
    $$ S_i = \lambda \times \mu_i + (1 - \lambda) \times \mu_{global} $$
    Where $\mu_i$ is the average price of area $i$, $\mu_{global}$ is the global average price.

#### d. Scaling
- Use **Min-Max Scaling** to bring all variables to the range $[0, 1]$. This is especially important for Lasso Regression as this algorithm is scale-sensitive.

### 2. Algorithms & Implementation

The project implements `LinearRegression` and `Lasso` classes inheriting a structure similar to Scikit-learn (`fit`, `predict`).

#### a. Linear Regression
**Goal:** Find the weight vector $\beta$ such that the Residual Sum of Squares (RSS) is minimized:
$$ J(\beta) = ||y - X\beta||_2^2 = \sum_{i=1}^{n} (y_i - x_i^T\beta)^2 $$

**Solution (Closed-form Solution):**
The optimal solution is calculated using the Normal Equation:
$$ \hat{\beta} = (X^T X)^{-1} X^T y $$

**Implementation with NumPy:**
Instead of calculating the matrix inverse $(X^T X)^{-1}$ (costly and unstable), we solve the linear equation system $A\beta = b$:
- Set $A = X^T X$ and $b = X^T y$.
- Use NumPy's optimization function:
  ```python
  # self.weights = np.linalg.inv(X.T @ X) @ X.T @ y  <-- Do not use
  self.weights = np.linalg.solve(X.T @ X, X.T @ y) # <-- More optimal
  ```

#### b. Lasso Regression (L1 Regularization)
**Goal:** Minimize the loss function with an added L1 regularization component (helps drive unimportant weights to 0):
$$ J(\beta) = \frac{1}{2n} ||y - X\beta||_2^2 + \alpha ||\beta||_1 $$

**Solution (Coordinate Descent):**
Since the L1 function is not differentiable at 0, we don't use standard Gradient Descent but **Coordinate Descent**. We optimize each weight $\beta_j$ while keeping weights $\beta_{k \neq j}$ fixed.

Update formula for $\beta_j$:
$$ \beta_j = S(\rho_j, \alpha) $$
Where:
- $\rho_j = \sum_{i=1}^{n} x_{ij} (y_i - \sum_{k \neq j} x_{ik}\beta_k)$ (correlation between variable $j$ and the residual).
- $S(z, \alpha)$ is the **Soft Thresholding** operator:
  $$ S(z, \alpha) = \begin{cases} z - \alpha & \text{if } z > \alpha \\ z + \alpha & \text{if } z < -\alpha \\ 0 & \text{if } |z| \le \alpha \end{cases} $$

**Implementation with NumPy:**
- Pre-compute constant values to speed up the loop.
- Use vectorization to calculate predictions $\hat{y}$ and residuals $r = y - \hat{y}$.
- Update weights iteratively until convergence (error change is smaller than threshold `tol`).

---

## ⚙️ Installation & Setup

### System Requirements
- Python 3.8+
- Libraries: NumPy, Matplotlib, Seaborn (see `requirements.txt`)

### Environment Setup
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Instructions for running each part
1.  **Data Exploration (EDA):**
    - Open and run `notebooks/01_data_exploration.ipynb` to view distribution analysis, maps, and correlations.
2.  **Preprocessing:**
    - Run `notebooks/02_preprocessing.ipynb` to perform cleaning, encoding, and scaling. Results will be saved to `data/processed/`.
3.  **Modeling & Evaluation:**
    - Run `notebooks/03_modeling.ipynb` to train Linear/Lasso models and view detailed results.

---

## 📈 Results

### 1. Achieved Metrics
Evaluation results on the Test set (20% of data) after parameter optimization:

| Metric | Train Set | Test Set | Cross-Validation (Mean) | Meaning |
|--------|-----------|----------|-------------------------|---------|
| **R² Score** | **0.554** | **0.546** | 0.553 (±0.011) | The model explains ~54.6% of the price variance. |
| **RMSE** | 0.462 | 0.468 | 0.462 (±0.011) | Root Mean Squared Error (on log scale). |
| **MAE** | 0.335 | 0.334 | 0.335 (±0.005) | Mean Absolute Error. |

> **Comment:** The R² difference between Train and Test sets is very small (0.008), indicating the model is **not Overfitting** and generalizes well.

### 2. Visualizations

#### a. Cross-Validation Scores
The chart shows the stability of the model across 5 data splits (5-fold CV).
![CV Scores](https://github.com/user-attachments/assets/51113648-c78a-4435-a21f-6ba823d677a7)

#### b. Actual vs Predicted
Scatter plot between actual and predicted prices. Points clustered around the red diagonal ($y=x$) indicate good accuracy, though the model tends to underpredict in the very high price segment (luxury).
![Actual vs Predicted](https://github.com/user-attachments/assets/a8467748-48d7-4d44-933c-4e896d531a01)

#### c. Residuals Analysis
The distribution of residuals is near-normal and centered around 0, indicating no major bias in the model.
![Residuals](https://github.com/user-attachments/assets/093617d2-fe72-4bdf-b19f-f790a976884e)

#### d. Feature Importance
Factors most strongly influencing room prices.
- **Price Increase:** Entire home/apt, Manhattan, expensive areas.
- **Price Decrease:** Shared room, Bronx, areas far from center.
![Feature Importance](https://github.com/user-attachments/assets/5798493d-649e-4207-a3d1-a0dadee32acd)

### 3. Comparison and Analysis

#### Feature Importance Analysis
Based on model weights ($\beta$), we draw important insights:

1.  **Location is paramount:**
    - `neighbourhood_group_Manhattan` has the largest positive coefficient (+4.266), confirming Manhattan as the most expensive area.
    - `dist_to_center` has a negative coefficient (-0.689), meaning the further from the center, the lower the price.
2.  **Room type determines price floor:**
    - `room_type` variables (Shared room, Private room) have very large negative coefficients compared to `Entire home/apt` (hidden in intercept or relative comparison), showing that renting an entire place is much more expensive than single rooms.
3.  **Availability:**
    - `availability_365` has a positive coefficient (+0.342), suggesting that professional listings (available year-round) often have higher prices than short-term/seasonal listings.

#### Linear Regression vs Lasso Comparison
- **Performance:** Both models yield similar results (R² ~ 0.55).
- **Selection:**
    - **Linear Regression** was chosen as the final model due to its simplicity and effectiveness without needing complex $\alpha$ parameter tuning.
    - **Lasso** is useful for identifying redundant features (driving coefficients to 0), but in this dataset, most selected features have statistical significance.

---

## 📂 Project Structure

```
├── data/
│   ├── raw/                # Raw data (AB_NYC_2019.csv)
│   └── processed/          # Processed data (train/test features)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: Distribution analysis, maps, correlations
│   ├── 02_preprocessing.ipynb     # Pipeline: Cleaning, Encoding, Scaling
│   └── 03_modeling.ipynb          # Modeling: Linear/Lasso from scratch, CV, Evaluation
├── src/
│   ├── data_processing.py  # Data processing utility functions
│   ├── models.py           # LinearRegression, Lasso, KFold class implementations
│   └── visualization.py    # Plotting functions
├── README.md               # Project documentation
└── requirements.txt        # Dependency list
```

---

## 🧩 Challenges & Solutions

### 1. Vectorization with NumPy
- **Challenge:** Converting mathematical formulas (like Coordinate Descent) from loops (for-loop) to vectorized forms to speed up computation on large datasets. Using loops in Python is very slow.
- **Solution:**
  - Fully utilize NumPy's **Broadcasting** to perform operations on entire arrays without loops.
  - Use optimized matrix operations (`@` for matrix multiplication, `np.sum`, `np.where`).
  - Pre-compute constant values (like $X^T X$ or $||x_j||^2$) outside the optimization loop.

### 2. Handling High Cardinality Categorical Variables
- **Challenge:** The `neighbourhood` column has over 200 distinct values. Using standard One-Hot Encoding would create over 200 new columns, making the feature matrix very sparse and increasing computational cost, while also risking overfitting.
- **Solution:** Apply **Target Encoding** combined with **Smoothing**. Instead of creating new columns, we replace `neighbourhood` values with the average target (`price`) for that area, adjusted (smoothed) to avoid noise in areas with little data.

### 3. Numerical Stability
- **Challenge:** When calculating the matrix inverse $(X^T X)^{-1}$ in Linear Regression, if the matrix $X^T X$ is near-singular or has a large condition number, the result will be very inaccurate.
- **Solution:** Instead of calculating the inverse directly with `np.linalg.inv`, use `np.linalg.solve` to solve the linear equation system, making the algorithm more stable and accurate.

---

## 🔮 Future Improvements

1.  **Model Expansion:**
    - Experiment with Non-linear models like **Decision Tree**, **Random Forest**, or **Gradient Boosting** (implemented from scratch) to capture complex relationships that linear models miss.
    - Implement **Ridge Regression** (L2 Regularization) and **Elastic Net** (combining L1 & L2).

2.  **Data Improvement:**
    - Integrate External Data such as: distance to nearest subway station, safety index, or distance to other famous tourist spots besides Times Square.
    - Use NLP to analyze review content or listing names (`name`) to extract more features (e.g., "luxury", "cozy", "view").

3.  **Application:**
    - Build a simple Web App (using Streamlit or Flask) allowing users to input apartment info and get instant price predictions.
    - Deploy an API to integrate into other systems.

---

## 👥 Contributors

**Shinoaki0145**
- **Role:** Data Scientist
- **Contact**: thtnhan23@clc.fitus.edu.vn
- **Github:** [Shinoaki0145](https://github.com/Shinoaki0145)

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
