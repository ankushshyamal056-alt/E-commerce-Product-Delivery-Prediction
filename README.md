# 📦 E-Commerce Product Delivery Prediction

A machine learning project that predicts whether an e-commerce shipment will be delivered **on time or late**, using historical customer behaviour and supply chain data.

---

## 📌 Project Summary

Late deliveries are a major pain point in e-commerce logistics. This project builds a classification model to forecast delivery timeliness, helping businesses identify the key drivers of delay and optimise their logistics strategies — ultimately improving operational efficiency and customer satisfaction.

---

## 🎯 Objective

> Develop a robust machine learning classification model that uses historical data to predict the probability of a shipment reaching the customer on time.

---

## 📂 Dataset

**File:** `E_Commerce.csv`  
**Size:** 10,999 observations × 12 features

| Category | Features |
|---|---|
| Logistics | Warehouse block, Mode of shipment, Weight (gms) |
| Customer Interaction | Customer care calls, Rating, Prior purchases |
| Product Info | Cost of product, Product importance, Discount offered |
| Demographics | Gender |
| **Target** | `Reached.on.Time_Y.N` — `0` = On Time, `1` = Late |

**Data Quality:**
- ✅ No missing values
- ✅ No duplicate rows
- 🗑️ Non-informative `ID` column dropped

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings from the analysis:

- **Target Distribution:** The dataset is slightly imbalanced, with the majority of shipments arriving late.
- **Discount offered** shows the strongest positive correlation with late delivery — large discounts are strongly associated with delays.
- **Heavy items** (>4,000 gms) are more likely to be delayed.
- **Flight shipments** have the highest proportion of late deliveries; Road and Ship modes perform better.
- **Warehouse Block B** shows a higher late delivery proportion compared to other blocks.
- **Gender** has no meaningful influence on delivery timeliness.
- **High-importance products** still experience significant delays despite their priority status.

---

## ⚙️ Data Preprocessing & Feature Engineering

| Step | Detail |
|---|---|
| Label Encoding | `Product_importance` (Low/Medium/High → 0/1/2), `Gender` (F/M → 0/1) |
| One-Hot Encoding | `Warehouse_block`, `Mode_of_Shipment` |
| Train/Test Split | 80% train, 20% test (`random_state=42`) |
| Feature Scaling | `StandardScaler` applied for KNN and Logistic Regression |

---

## 🤖 Models Trained

Four classification algorithms were evaluated:

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline linear classifier |
| **Decision Tree** | Rule-based, interpretable tree structure |
| **Random Forest** | Ensemble of 100 decision trees |
| **K-Nearest Neighbors (KNN)** | Distance-based, uses k=5 |

Logistic Regression and KNN were trained on **scaled** features; Decision Tree and Random Forest on **unscaled** features.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~63% |
| Decision Tree | ~63% |
| **Random Forest** ⭐ | **~66%** |
| KNN | ~63% |

**🏆 Winner: Random Forest Classifier**  
- **Accuracy:** 66%  
- **F1-Score:** 0.69  
- Best at capturing non-linear relationships between features like discount and weight.

---

## 🔑 Feature Importance (Random Forest)

Top features driving delivery predictions:

1. **Discount Offered** — By far the most influential predictor
2. **Weight in grams** — Heavier items are harder to deliver on time
3. **Cost of the Product** — Higher-cost items show different delivery patterns
4. **Prior Purchases & Customer Care Calls** — Moderate influence
5. **Gender, Warehouse Block, Mode of Shipment** — Minimal impact

---

## 💡 Recommendations

- **High-Importance Products:** Investigate why priority items still face frequent delays — special handling procedures may be creating bottlenecks.
- **Discount Strategy:** Avoid offering high discounts on already high-risk shipments (e.g., heavy items shipped by air), as large discounts strongly correlate with delays.
- **Shipment Mode:** Shift non-urgent, heavy, or heavily discounted items from air (Flight) to Road or Ship transport to cut costs and improve reliability.

---

## 🛠️ Libraries Used

```
pandas | numpy | matplotlib | seaborn | scikit-learn
```

Specifically from `scikit-learn`:
- `train_test_split`, `LabelEncoder`, `StandardScaler`
- `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`
- `accuracy_score`, `classification_report`, `confusion_matrix`

---

## 🚀 Getting Started

1. **Clone the repository** and ensure `E_Commerce.csv` is in the project root.
2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Run the notebook:**
   ```bash
   jupyter notebook E-commerce_Product_Delivery_Prediction.ipynb
   ```

---

## 📁 Project Structure

```
├── E-commerce_Product_Delivery_Prediction.ipynb   # Main notebook
├── E_Commerce.csv                                  # Dataset
└── README.md                                       # This file
```
