Florida Housing Market Analysis & Forecasting

This project explores the relationship between **Florida's population growth**, **new housing unit construction**, and **housing prices** over time. It uses public datasets to clean, merge, visualize, and model trends generating **forecasted housing prices** up to 2040 using linear and polynomial regression.

[Live Dashboard} (https://floridahousingprojections.streamlit.app/)
---

## Datasets Used

1. **FLSTHPI.csv** – Housing price index data for Florida  
2. **NewHousingUnits.csv** – Monthly new housing units constructed  
3. **statepop.xlsx** – Florida’s yearly population figures

---

## Project Goals

- Analyze how housing prices in Florida are influenced by:
  - Population growth
  - Housing unit supply
- Visualize yearly trends
- Apply **linear** and **polynomial regression** to predict housing prices
- Forecast housing price trends through **2040**

---

## Features

- **Data Preprocessing**: Date parsing, missing value checks, merging on `Year`
- **Visualization**: Matplotlib plots showing:
  - New housing units over time
  - Average housing prices
  - Population trends
  - Housing prices vs population
- **ML Models**:
  - Linear Regression (Normal Equation)
  - Polynomial Regression (Degree 2)
-  **Forecasting**:
  - Predict future housing prices based on extrapolated population and housing trends

---

[image](https://github.com/user-attachments/assets/27dd3160-515f-4bf9-8b20-31cf3940c887)

[image](https://github.com/user-attachments/assets/2acabf17-c9f5-4ce6-9a48-f11cb171422d)



---


## Technologies Used

-Python
-Streamlit
-Pandas,NumPy
-Matplotlib
-scikit-learn
