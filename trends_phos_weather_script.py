# Importing required packages
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import json

# Importing phosphorus dataset
phosphorus = pd.read_excel("phosphorus-lakes.xlsx", na_values=["..."])

# Importing weather dataset, each sheet being a distinct dataframe
insolation = pd.read_excel("weather-data.xlsx", sheet_name="Insolation", na_values=["..."])
rainfall = pd.read_excel("weather-data.xlsx", sheet_name="Rainfall", na_values=["..."])
temperature = pd.read_excel("weather-data.xlsx", sheet_name="Temperature", na_values=["..."])
snow = pd.read_excel("weather-data.xlsx", sheet_name="Fresh snow", na_values=["..."])

# Function to generate descriptive statistics
def desc_stat(df, x_col, y_col):
    stats = df[[x_col, y_col]].iloc[:, 1:].describe().to_dict()
    return stats

# Function for analysing linear trends
def detect_trends(df, x_col, y_col):
    # Removing NaN values for both columns
    valid_data = df[[x_col, y_col]].dropna()
    
    # Making sure there is enough values for the analysis
    if len(valid_data) > 1:
        x = valid_data[x_col].astype(float)
        y = valid_data[y_col].astype(float)
        regression = linregress(x, y)
        trend = "positive" if regression.slope > 0 else "negative"
        return {"slope": regression.slope, "intercept": regression.intercept, "r_value": regression.rvalue, "trend": trend}
    else:
        return None

# Function for plotting graphs
def plot_graph(dataframes, x_col, y_cols, title, xlab, xlim_inf, xlim_sup, ylab):
    plt.figure(figsize=(12, 8))

    cmap = plt.get_cmap("tab20") # Use color map with 20 different colors
    num_colors = cmap.N # Get the number of colors in the color map

    for i, (df, y_col, label) in enumerate(dataframes):
        color = cmap(i % num_colors) # Cycle through the color map
        plt.plot(df[x_col], df[y_col], marker = "o", linestyle = "-", label = label, color = color, alpha = 0.8)
    
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.xlim(xlim_inf,xlim_sup)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left") # Avoid overlapping with the graph
    plt.tight_layout()
    plt.show()

# Function for plotting interactive graphs
def interactive_plot(df, x_col, y_cols, title, xlab, xlim_inf, xlim_sup, ylab):
    cmap = plt.get_cmap("tab20")
    num_colors = cmap.N

    # Widget to select stations dynamically
    station_selector = widgets.SelectMultiple(
        options = y_cols,
        value = tuple(y_cols[:3]), # Default: first 3 stations selected
        description = "Stations:",
        style = {"description_width": "initial"}
    )

    # Function to update dynamically the plot
    def update_plot(selected_stations):
        plt.figure(figsize=(12, 8))

        for i, station in enumerate(selected_stations):
            color = cmap(i % num_colors)
            plt.plot(df[x_col], df[station], marker = "o", linestyle = "-", label = station, color = color, alpha = 0.8)
        
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True)
        plt.xlim(xlim_inf, xlim_sup)
        plt.legend(bbox_to_anchor = (1.05,1), loc = "upper left")
        plt.tight_layout()
        plt.show()
    
    # Create an interactive widget
    widgets.interactive(update_plot, selected_stations = station_selector)

    # Display the widget
    display(station_selector)
    update_plot(station_selector.value) # Show the intial plot

# Function for comparing trends
def compare_trends(trend1, trend2):
    if trend1["trend"] == trend2["trend"]:
        return "Trends are similar."
    else:
        return "Trends are opposite."

# Trends detection for each dataset
trends_results = {}
descriptive_stats = {}
comparisons = {}

# Phosphorus analysis
lakes = phosphorus.columns[1:]
data_to_plot = []
trends_results["phosphorus"] = {}
descriptive_stats["phosphorus"] = {}
for lake in lakes:
    trend = detect_trends(phosphorus, "Year", lake)
    stats = desc_stat(phosphorus, "Year", lake)
    if trend:
        trends_results["phosphorus"][lake] = trend
        descriptive_stats["phosphorus"][lake] = stats
        data_to_plot.append((phosphorus, lake, lake))
plot_graph(data_to_plot, "Year", lakes, "Phosphorus trend", "Years", 1952, 2028, "Water phosphorus concentration (µg/l)")

# Weather analysis
for dataset_name, dataset in [("insolation", insolation), ("rainfall", rainfall), ("temperature", temperature), ("snow", snow)]:
    stations = dataset.columns[1:]
    trends_results[dataset_name] = {}
    descriptive_stats[dataset_name] = {}

    for station in stations:
        trend = detect_trends(dataset, "Year", station)
        stats = desc_stat(dataset, "Year", station)

        if trend:
            trends_results[dataset_name][station] = trend
            descriptive_stats[dataset_name][station] = stats

# Interactive plot for weather datas
# Insolation
interactive_plot(insolation, "Year", insolation.columns[1:], "Insolation trend", "Years", 1926, 2029, "Annual insolation time (hour)")

# Rainfall
interactive_plot(rainfall, "Year", rainfall.columns[1:], "Rainfall trend", "Years", 1926, 2029, "Annual rainfall (mm)")

# Temperature
interactive_plot(temperature, "Year", temperature.columns[1:], "Temperature trend", "Years", 1926, 2029, "Annual mean temperature (°C)")

# Fresh snow
interactive_plot(snow, "Year", snow.columns[1:], "Fresh snow trend", "Years", 1926, 2029, "Annual fresh snow (cm)")

# Comparing trends
for lake, phos_trend in trends_results["phosphorus"].items():
    for climate_type in ["insolation", "rainfall", "temperature", "snow"]:
        for station, climate_trend in trends_results[climate_type].items():
            comparison = compare_trends(phos_trend, climate_trend)
            comparisons[f"{lake} vs {station} ({climate_type})"] = comparison

for climate_type1 in ["insolation", "rainfall", "temperature", "snow"]:
    for station1, trend1 in trends_results[climate_type1].items():
        for climate_type2 in ["insolation", "rainfall", "temperature", "snow"]:
            if climate_type1 != climate_type2:
                for station2, trend2 in trends_results[climate_type2].items():
                    comparison = compare_trends(trend1, trend2)
                    comparisons[f"{station1} ({climate_type1}) vs {station2} ({climate_type2})"] = comparison

# Saving trends results
with open("trends_results.json", "w") as f:
    json.dump({"Descriptive statistics": descriptive_stats, "Trends analysis results": trends_results, "Trends comparison": comparisons}, f, indent=4)

print("Trends analysis completed. The results are saved in 'trends_results.json'.")
