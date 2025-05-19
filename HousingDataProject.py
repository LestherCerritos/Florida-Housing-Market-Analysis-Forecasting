import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np





def load_and_process_housing_prices():
    df = pd.read_csv("FLSTHPI.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Year"] = df["DATE"].dt.year
    yearly = df.groupby("Year")["FLSTHPI"].mean().reset_index()
    yearly.rename(columns={"FLSTHPI": "Housing_Price_Index"}, inplace=True)
    return df, yearly

def load_and_process_new_units():
    df = pd.read_csv("NewHousingUnits.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Year"] = df["DATE"].dt.year
    yearly = df.groupby("Year")["FLBPPRIV"].sum().reset_index()
    yearly.rename(columns={"FLBPPRIV": "New_Housing_Units"}, inplace=True)
    return df, yearly

def load_and_process_population():
    df = pd.read_excel("statepop.xlsx")
    df["Population_Growth"] = df["Population"].pct_change() * 100
    return df

new_housing_prices, housing_prices_yearly = load_and_process_housing_prices()
new_units_data, housing_units_yearly = load_and_process_new_units()
population_data = load_and_process_population()

def merge_all_data(pop_df, housing_df, units_df):
    merged = pd.merge(pop_df, housing_df, on="Year", how="inner")
    merged = pd.merge(merged, units_df, on="Year", how="inner")
    merged.rename(columns={
        "Population": "Population",
        "Population_Growth": "Population Growth (%)",
        "Housing_Price_Index": "Average Housing Price",
        "New_Housing_Units": "New Housing Units",
    }, inplace=True)
    return merged

merged_data = merge_all_data(population_data, housing_prices_yearly, housing_units_yearly)
def debug_info(population_data, housing_units_yearly, housing_prices_yearly, merged_data, new_housing_prices, new_units_data):
    st.subheader("🧪 Debug Information")

    st.markdown("### Merged Dataset Preview")
    st.dataframe(merged_data)

    st.markdown("### Missing Values in Merged Dataset")
    st.dataframe(merged_data.isnull().sum())

    st.markdown("### Yearly Aggregated New Housing Units Data")
    st.dataframe(housing_units_yearly)

    st.markdown("### Population Data with Growth Rates")
    st.dataframe(population_data)

    # Year ranges
    st.markdown("### Year Ranges in Datasets")
    st.text(f"Population Data: {population_data['Year'].min()} - {population_data['Year'].max()}")
    st.text(f"Housing Prices Data: {housing_prices_yearly['Year'].min()} - {housing_prices_yearly['Year'].max()}")
    st.text(f"New Housing Units Data: {housing_units_yearly['Year'].min()} - {housing_units_yearly['Year'].max()}")

    # Unhealthy rows removed
    unhealthy_housing = len(new_housing_prices) - len(housing_prices_yearly)
    unhealthy_units = len(new_units_data) - len(housing_units_yearly)

    st.markdown("### Unhealthy Rows Removed")
    st.text(f"Housing Data: {unhealthy_housing}")
    st.text(f"New Housing Units Data: {unhealthy_units}")
    st.text(f"Population Data: 0")

    st.markdown("### Final Merged Dataset Size")
    st.text(f"Total Rows: {len(merged_data)}")


#GRAPHS

def plot_new_housing_units(housing_units_yearly):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(housing_units_yearly["Year"], housing_units_yearly["New_Housing_Units"],
            marker="o", linestyle="-", label="New Housing Units")
    ax.set_title("Yearly New Housing Units in Florida (1988–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("New Housing Units")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

def plot_housing_prices(housing_prices_yearly):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(housing_prices_yearly["Year"], housing_prices_yearly["Housing_Price_Index"],
            marker="o", linestyle="-", label="Housing Prices")
    ax.set_title("Yearly Average Housing Prices in Florida (2000–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Housing Prices (ZHVI)")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

def plot_population(population_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(population_data["Year"], population_data["Population"],
            marker="o", linestyle="-", label="Population")
    ax.set_title("Population in Florida Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

def plot_population_vs_prices(merged_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(merged_data["Population"], merged_data["Average Housing Price"],
               color="blue", label="Population vs Housing Prices")
    ax.set_title("Population vs. Housing Prices")
    ax.set_xlabel("Population")
    ax.set_ylabel("Average Housing Price")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

def display_static_charts(housing_prices_yearly, housing_units_yearly, population_data, merged_data):
    st.subheader("🏗️ New Housing Units Over Time")
    plot_new_housing_units(housing_units_yearly)

    st.subheader("💰 Housing Prices Over Time")
    plot_housing_prices(housing_prices_yearly)

    st.subheader("👥 Population Over Time")
    plot_population(population_data)

    st.subheader("📈 Population vs Housing Prices")
    plot_population_vs_prices(merged_data)


# ─── MODELS & PLOTS ─────────────────────────────

def forecast_models(merged_data):
    # Prepare features and target
    X = merged_data[["Population", "Population Growth (%)", "New Housing Units"]].values
    y = merged_data["Average Housing Price"].values

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear model
    X_train_lin = np.column_stack((np.ones(len(X_train)), X_train))
    theta_lin = np.linalg.lstsq(X_train_lin, y_train, rcond=None)[0]

    # Polynomial model
    X_train_poly = np.column_stack((X_train, X_train**2))
    X_train_poly = np.column_stack((np.ones(len(X_train_poly)), X_train_poly))
    theta_poly = np.linalg.lstsq(X_train_poly, y_train, rcond=None)[0]

    # Predictions over existing years
    merged_sorted = merged_data.sort_values("Year")
    years = merged_sorted["Year"].values
    X_full = merged_sorted[["Population", "Population Growth (%)", "New Housing Units"]].values

    X_full_lin = np.column_stack((np.ones(X_full.shape[0]), X_full))
    y_pred_lin_full = X_full_lin @ theta_lin

    X_full_poly = np.column_stack((X_full, X_full**2))
    X_full_poly = np.column_stack((np.ones(X_full_poly.shape[0]), X_full_poly))
    y_pred_poly_full = X_full_poly @ theta_poly

    # Future feature projections
    years_hist = merged_data["Year"].values
    pop_hist = merged_data["Population"].values
    units_hist = merged_data["New Housing Units"].values
    growth_hist = merged_data["Population Growth (%)"].values

    pop_coef = np.polyfit(years_hist, pop_hist, 1)
    units_coef = np.polyfit(years_hist, units_hist, 1)
    growth_coef = np.polyfit(years_hist, growth_hist, 1)

    future_years = np.arange(years_hist.max()+1, 2041)
    pop_fut = np.polyval(pop_coef, future_years)
    units_fut = np.polyval(units_coef, future_years)
    growth_fut = np.polyval(growth_coef, future_years)

    X_fut = np.column_stack((pop_fut, growth_fut, units_fut))

    # Future predictions
    X_fut_lin = np.column_stack((np.ones(X_fut.shape[0]), X_fut))
    y_fut_lin = X_fut_lin @ theta_lin

    X_fut_poly = np.column_stack((X_fut, X_fut**2))
    X_fut_poly = np.column_stack((np.ones(X_fut_poly.shape[0]), X_fut_poly))
    y_fut_poly = X_fut_poly @ theta_poly

    return years, y_pred_lin_full, y_pred_poly_full, future_years, y_fut_lin, y_fut_poly

def plot_forecast_models(years, y_lin, y_poly, future_years, y_fut_lin, y_fut_poly):
    # Historical prediction
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(years, y_lin, marker="o", label="Linear Predictions")
    ax1.plot(years, y_poly, marker="s", linestyle="--", label="Polynomial Predictions")
    ax1.set_title("Predicted Housing Prices (Historical)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Predicted Housing Price Index")
    ax1.grid()
    ax1.legend()
    st.pyplot(fig1)

    # Forecast
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(future_years, y_fut_lin, marker='o', linestyle='-', label="Linear Forecast")
    ax2.plot(future_years, y_fut_poly, marker='s', linestyle='--', label="Polynomial Forecast")
    ax2.set_title("Forecasted Florida Housing Prices to 2040")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Predicted Housing Price Index")
    ax2.grid()
    ax2.legend()
    st.pyplot(fig2)

def run_dashboard():
    # ─── PAGE SETTINGS ─────────────────────────────
    st.set_page_config(
        page_title="Florida Housing Market Dashboard",
        layout="wide"
    )
    st.title("🏠 Florida Housing Market Dashboard")

    # ─── LOAD & PROCESS DATA ───────────────────────
    new_housing_prices, housing_prices_yearly = load_and_process_housing_prices()
    new_units_data, housing_units_yearly = load_and_process_new_units()
    population_data = load_and_process_population()
    merged_data = merge_all_data(population_data, housing_prices_yearly, housing_units_yearly)

    # ─── STATIC CHARTS ─────────────────────────────
    st.header("📊 Historical Housing & Population Trends")
    display_static_charts(
        housing_prices_yearly,
        housing_units_yearly,
        population_data,
        merged_data
    )

    # ─── FORECAST CHARTS ───────────────────────────
    st.header("🔮 Housing Price Forecasting")
    years, y_lin, y_poly, future_years, y_fut_lin, y_fut_poly = forecast_models(merged_data)
    plot_forecast_models(years, y_lin, y_poly, future_years, y_fut_lin, y_fut_poly)

    # ───  DEBUG INFO ───────────────────────
    with st.expander("🛠 Show Debug Info"):
        debug_info(
            population_data,
            housing_units_yearly,
            housing_prices_yearly,
            merged_data,
            new_housing_prices,
            new_units_data
        )





if __name__ == "__main__":
    run_dashboard()
