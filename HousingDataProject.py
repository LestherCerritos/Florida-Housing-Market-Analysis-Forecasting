import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


# Load datasets
new_housing_prices = pd.read_csv("FLSTHPI.csv")
new_units_data = pd.read_csv("NewHousingUnits.csv")
population_data = pd.read_excel("statepop.xlsx")

# Process the New Housing Prices Data
new_housing_prices["DATE"] = pd.to_datetime(new_housing_prices["DATE"], errors="coerce")
new_housing_prices["Year"] = new_housing_prices["DATE"].dt.year

# Aggregate housing price data by year
housing_prices_yearly = (
    new_housing_prices.groupby("Year")["FLSTHPI"]
    .mean()
    .reset_index()
    .rename(columns={"FLSTHPI": "Housing_Price_Index"})
)

# Debug: Print the yearly housing prices
print("Yearly Aggregated Housing Prices:")
print(housing_prices_yearly)






# Process New Housing Units Data
new_units_data["DATE"] = pd.to_datetime(new_units_data["DATE"], errors="coerce")
new_units_data["Year"] = new_units_data["DATE"].dt.year
housing_units_yearly = new_units_data.groupby("Year")["FLBPPRIV"].sum().reset_index()
housing_units_yearly.rename(columns={"FLBPPRIV": "New_Housing_Units"}, inplace=True)

# Process Population Data
population_data["Population_Growth"] = population_data["Population"].pct_change() * 100

# Merge datasets on Year
merged_data = pd.merge(population_data, housing_prices_yearly, on="Year", how="inner")
merged_data = pd.merge(merged_data, housing_units_yearly, on="Year", how="inner")

# Rename columns for clarity
merged_data.rename(
    columns={
        "Population": "Population",
        "Population_Growth": "Population Growth (%)",
        "Housing_Price_Index": "Average Housing Price",
        "New_Housing_Units": "New Housing Units",
    },
    inplace=True,
)

# Debug: Print merged dataset
print("Merged Dataset Preview:")
print(merged_data)

# Check for missing values
print("Missing Values in Merged Dataset:")
print(merged_data.isnull().sum())


# Convert DATE column to datetime
new_units_data["DATE"] = pd.to_datetime(new_units_data["DATE"], errors="coerce")



# Check unique years in the DATE column
new_units_data["Year"] = new_units_data["DATE"].dt.year


# Aggregate the data by year
housing_units_yearly = new_units_data.groupby("Year")["FLBPPRIV"].sum().reset_index()
housing_units_yearly.rename(columns={"FLBPPRIV": "New_Housing_Units"}, inplace=True)

print("Yearly Aggregated New Housing Units Data:")
print(housing_units_yearly)



# Calculate population growth
population_data["Population_Growth"] = population_data["Population"].pct_change() * 100

# Check the results
print("Population Data with Growth Rates:")
print(population_data)


florida_housing_yearly = housing_prices_yearly




# Check year ranges in each dataset
print("Year Range in Population Data:", population_data["Year"].min(), "-", population_data["Year"].max())
print("Year Range in Housing Prices Data:", florida_housing_yearly["Year"].min(), "-", florida_housing_yearly["Year"].max())
print("Year Range in New Housing Units Data:", housing_units_yearly["Year"].min(), "-", housing_units_yearly["Year"].max())

# Merge population and housing data
merged_data = pd.merge(population_data, florida_housing_yearly, on="Year", how="inner")

# Merge the result with new housing units data
merged_data = pd.merge(merged_data, housing_units_yearly, on="Year", how="inner")

# Rename columns for clarity
merged_data.rename(columns={
    "Population": "Population",
    "Population_Growth": "Population Growth (%)",
    "Housing_Price": "Average Housing Price",
    "New_Housing_Units": "New Housing Units"
}, inplace=True)

# Display the merged dataset
print("Merged Dataset Preview:")
print(merged_data)

# Check for missing values
print("Missing Values in Merged Dataset:")
print(merged_data.isnull().sum())



# Count initial and final rows
initial_housing_count = len(new_housing_prices)
initial_units_count = len(new_units_data)
initial_population_count = len(population_data)

final_housing_count = len(florida_housing_yearly)
final_units_count = len(housing_units_yearly)
final_population_count = len(population_data)

# Count unhealthy rows
unhealthy_housing = initial_housing_count - final_housing_count
unhealthy_units = initial_units_count - final_units_count
unhealthy_population = initial_population_count - final_population_count

print("Unhealthy Rows Removed:")
print(f"Housing Data: {unhealthy_housing}")
print(f"New Housing Units Data: {unhealthy_units}")
print(f"Population Data: {unhealthy_population}")


print("Number of instances in the final merged dataset:", len(merged_data))





#GRAPHS

# Plot yearly aggregated new housing units
plt.figure(figsize=(12, 6))
plt.plot(housing_units_yearly["Year"], housing_units_yearly["New_Housing_Units"], marker="o", linestyle="-", label="New Housing Units")
plt.title("Yearly New Housing Units in Florida (1988-2024)")
plt.xlabel("Year")
plt.ylabel("New Housing Units")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    florida_housing_yearly["Year"],
    florida_housing_yearly["Housing_Price_Index"],  # ← fixed name here
    marker="o",
    linestyle="-",
    label="Housing Prices"
)
plt.title("Yearly Average Housing Prices in Florida (2000-2024)")
plt.xlabel("Year")
plt.ylabel("Housing Prices (ZHVI)")
plt.grid()
plt.legend()
plt.show()


# Plot population over the years
plt.figure(figsize=(12, 6))
plt.plot(population_data["Year"], population_data["Population"], marker="o", linestyle="-", label="Population")
plt.title("Population in Florida Over Time")
plt.xlabel("Year")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()




# Plot population vs. housing prices
plt.figure(figsize=(12, 6))
plt.scatter(merged_data["Population"], merged_data["Housing_Price_Index"], color="blue", label="Population vs Housing Prices")
plt.title("Population vs. Housing Prices")
plt.xlabel("Population")
plt.ylabel("Average Housing Price")
plt.grid()
plt.legend()
plt.show()





# ─── MODELS & PLOTS ─────────────────────────────

# 1. Prepare features & target using the real column name
X = merged_data[["Population", "Population Growth (%)", "New Housing Units"]].values
y = merged_data["Housing_Price_Index"].values

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit linear regression (normal equation)
X_train_lin = np.column_stack((np.ones(len(X_train)), X_train))
theta = np.linalg.lstsq(X_train_lin, y_train, rcond=None)[0]

# 4. Fit polynomial regression (degree 2)
X_train_poly = np.column_stack((X_train, X_train**2))
X_train_poly = np.column_stack((np.ones(len(X_train_poly)), X_train_poly))
theta_poly = np.linalg.lstsq(X_train_poly, y_train, rcond=None)[0]



# 5) Prepare full‐range predictions over every Year in merged_data
merged_sorted = merged_data.sort_values("Year")
years = merged_sorted["Year"].values
X_full = merged_sorted[["Population", "Population Growth (%)", "New Housing Units"]].values

# 5a) Linear full predictions
X_full_lin = np.column_stack((np.ones(X_full.shape[0]), X_full))
y_pred_lin_full = X_full_lin @ theta

# 5b) Polynomial full predictions
X_full_poly = np.column_stack((X_full, X_full**2))
X_full_poly = np.column_stack((np.ones(X_full_poly.shape[0]), X_full_poly))
y_pred_poly_full = X_full_poly @ theta_poly

# 6) Plot Linear Predictions over time
plt.figure(figsize=(12, 6))
plt.plot(years, y_pred_lin_full,
         marker="o", linestyle="-", label="Linear Predictions")
plt.title("Linear Regression: Predicted Housing Prices Over Time")
plt.xlabel("Year")
plt.ylabel("Predicted Housing Price Index")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 7) Plot Polynomial Predictions over time
plt.figure(figsize=(12, 6))
plt.plot(years, y_pred_poly_full,
         marker="o", linestyle="-", label="Polynomial Predictions")
plt.title("Polynomial Regression: Predicted Housing Prices Over Time")
plt.xlabel("Year")
plt.ylabel("Predicted Housing Price Index")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# historical arrays
years_hist  = merged_data["Year"].values
pop_hist    = merged_data["Population"].values
units_hist  = merged_data["New Housing Units"].values
growth_hist = merged_data["Population Growth (%)"].values

# fit simple linear trends
pop_coef    = np.polyfit(years_hist, pop_hist,    deg=1)
units_coef  = np.polyfit(years_hist, units_hist,  deg=1)
growth_coef = np.polyfit(years_hist, growth_hist, deg=1)

# future year range
future_years = np.arange(years_hist.max()+1, 2041)

# extrapolate features
pop_fut    = np.polyval(pop_coef,    future_years)
units_fut  = np.polyval(units_coef,  future_years)
growth_fut = np.polyval(growth_coef, future_years)

# stack into feature matrix
X_fut = np.column_stack((pop_fut, growth_fut, units_fut))

# 5a) Linear model forecast
X_fut_lin      = np.column_stack((np.ones(X_fut.shape[0]), X_fut))
y_fut_lin      = X_fut_lin @ theta

# 5b) Polynomial model forecast (degree 2)
X_fut_poly     = np.column_stack((X_fut, X_fut**2))
X_fut_poly     = np.column_stack((np.ones(X_fut_poly.shape[0]), X_fut_poly))
y_fut_poly     = X_fut_poly @ theta_poly

# ─── 6) PLOT FORECASTS ────────────────────────────────────────────────────────

plt.figure(figsize=(12, 6))
plt.plot(future_years, y_fut_lin,  marker='o', linestyle='-',  label="Linear Forecast")
plt.plot(future_years, y_fut_poly, marker='s', linestyle='--', label="Polynomial Forecast")

plt.title("Forecasted Florida Housing Prices to 2040")
plt.xlabel("Year")
plt.ylabel("Predicted Housing Price Index")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()