# Importing required modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Loading each sheet into a dictionnary of dataframe
xls_path = "quality-of-life-trends.xlsx"
sheets = pd.read_excel(xls_path, sheet_name=None)

# Access each sheet by its name
total = sheets["Total"]
sex = sheets["Sex"]
age = sheets["Age"]
language = sheets["Language"]
city_country = sheets["City-country"]
education = sheets["Education"]
nationality = sheets["Nationality"]
economy = sheets["Economy"]


#for sheet_name, df in sheets.items():
#    print(df.head())


# Converting object data into float data
for sheet_name, df in sheets.items():
    for col in df.columns:
        if col not in ["category", "factor"]: # Ignoring the two first text columns
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                print(f"Error in {sheet_name}, column {col}: {e}")


# Checking data type in columns
#for sheet_name, df in sheets.items():
#    print(f"--- {sheet_name} ---")
#    print(df.dtypes, "\n")


# Checking any missing values in the dataframes
#for sheet_name, df in sheets.items():
#    missing_values = df.isna().sum().sum() # Total number of NaN
#    print(f"Sheet '{sheet_name}': {missing_values} missing values")


# Descriptive statistics of each dataframe
for sheet_name, df in sheets.items():
    print(f"--- Statistics for {sheet_name} ---")
    print(df.describe())
    print("\n")


# Histogram for 'total' sheet
df_total = sheets["Total"]

labels = df_total["factor"]
responses = ["very important", "rather important", "rather not important", "not at all important"]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

colors = ["blue", "green", "orange", "red"]
for i, response in enumerate(responses):
    ax.bar(x + i * width, df_total[response], width, label=response, color=colors[i])

ax.set_xticks(x + width * (len(responses) / 2 - 0.5))  # Center labels
ax.set_xticklabels(labels, rotation=45, ha="right")  # Display factors (45°)
ax.set_title("Distribution of answers by factor of quality of life")
ax.set_xlabel("Factor of quality of life")
ax.set_ylabel("Percentage of answer")
ax.legend(title="Importance")

plt.tight_layout()
plt.show()


# Test heatmap for 'sex' sheet
sex = sex.pivot(index="factor", columns="category", values="very important")

plt.figure(figsize=(10, 6))
sns.heatmap(sex, annot=True, cmap="coolwarm", linewidth=0.5)

plt.title("Importance of factors of quality of life")
plt.xlabel("Factors")
plt.ylabel("Categories")

plt.show()
