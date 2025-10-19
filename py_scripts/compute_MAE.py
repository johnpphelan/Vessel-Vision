#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load the Excel file
file_path = "boat_detections_sample_preserved_difference.xlsx"
df = pd.read_excel(file_path)

# Quick check of columns
print(df.columns)

# Compute basic metrics
# Mean Absolute Error (MAE)
df['Difference'] = df['Boat_Checked'] - df['boat_count']
mae = df['Difference'].abs().mean()
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Exact count accuracy (how often the model predicted exactly the correct number of boats)
exact_accuracy = (df['Difference'] == 0).mean()
print(f"Exact count accuracy: {exact_accuracy:.2%}")

# Optional: Overcount and undercount statistics
overcount = (df['Difference'] > 0).sum()
undercount = (df['Difference'] < 0).sum()
print(f"Overcount cases: {overcount}, Undercount cases: {undercount}")
#%%
# Compute difference just in case
df['Difference'] = df['Boat_Checked'] - df['boat_count']

# --- Scatter plot: True vs Predicted ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='boat_count', y='Boat_Checked', data=df, s=60)
plt.plot([df['boat_count'].min(), df['boat_count'].max()],
         [df['boat_count'].min(), df['boat_count'].max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("True Boat Count")
plt.ylabel("Predicted Boat Count")
plt.title("Boat Detection: True vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# --- Histogram of differences ---
plt.figure(figsize=(8,5))
sns.histplot(df['Difference'], bins=range(int(df['Difference'].min())-1, int(df['Difference'].max())+2), kde=False)
plt.xlabel("Prediction Error (Predicted - True)")
plt.ylabel("Number of Images")
plt.title("Distribution of Prediction Errors")
plt.grid(axis='y')
plt.show()
#%%