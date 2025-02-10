
# Author: Bini Chandra
# Date: 02/03/2025
# This python file contains the code written for HW3.

# Importing necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
t1 = pd.read_csv("Data/t1_user_active_min.csv")
t2 = pd.read_csv("Data/t2_user_variant.csv")

################################################## PART 2 #########################################################

# Merge datasets on 'uid' (ignoring 'dt' from t2_variant since it's always 2019-02-06)
merged_data = t1.merge(t2[['uid', 'variant_number']], on='uid', how='inner')

print("\nMerged Dataset:\n")
print(merged_data.head())


################################################## PART 3 #########################################################

# Plot Histogram
def plot_histogram(control, treatment, label1="Control Group", label2="Treatment Group"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(control, bins=30, kde=True, color="blue", alpha=0.6)
    plt.title(f"Histogram of Active Minutes ({label1})")
    plt.xlabel("Active Minutes")
    
    plt.subplot(1, 2, 2)
    sns.histplot(treatment, bins=30, kde=True, color="red", alpha=0.6)
    plt.title(f"Histogram of Active Minutes ({label2})")
    plt.xlabel("Active Minutes")
    
    plt.show()

# Plot QQPlot
def plot_qqplot(control, treatment, label1="Control Group", label2="Treatment Group"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(control, dist="norm", plot=ax[0])
    ax[0].get_lines()[0].set_marker('o')
    ax[0].get_lines()[0].set_markersize(3)
    ax[0].get_lines()[0].set_alpha(0.5)
    ax[0].set_title(f"QQ Plot ({label1})")
    
    stats.probplot(treatment, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set_marker('o')
    ax[1].get_lines()[0].set_markersize(3)
    ax[1].get_lines()[0].set_alpha(0.5)
    ax[1].set_title(f"QQ Plot ({label2})")
    
    plt.show()


# Separate Control and Treatment groups
control_group = merged_data[merged_data["variant_number"] == 0]["active_mins"]
treatment_group = merged_data[merged_data["variant_number"] == 1]["active_mins"]

# Plot Histograms to check Normality
plot_histogram(control_group, treatment_group)

# Create QQ Plots to check Normality
plot_qqplot(control_group, treatment_group)

# Compute Mean, Median, Standard Deviation and Variance
grouped_stats = merged_data.groupby("variant_number")["active_mins"].agg(["mean", "median", "std", "var"])
print(grouped_stats)

# Since data is NOT normal, we perform Mann-Whitney U-Test
u_test = stats.mannwhitneyu(control_group, treatment_group)
print("\nMann-Whitney U-Test Results:")
print(u_test)



################################################## PART 4 #########################################################

# Plot Histogram
def plot_boxplot(control, treatment, label1="Control Group", label2="Treatment Group"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=control, color="blue")
    plt.title(f"Boxplot of Active Minutes ({label1})")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=treatment, color="red")
    plt.title(f"Boxplot of Active Minutes ({label2})")
    
    plt.show()

# Remove outliers from the data
def remove_outliers_iqr(df, column):
    """Removes outliers using the IQR method."""
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Plot Boxplot of Control and Treatment groups
plot_boxplot(control_group, treatment_group)

# Remove outliers above 1,440 minutes
merged_clean = remove_outliers_iqr(merged_data, "active_mins")

# Display how many outliers were removed
print(f"\nOutliers removed: {len(merged_data) - len(merged_clean)}")
print("Max Active Minutes After Outlier Removal:", merged_clean["active_mins"].max())

# Separate Control and Treatment groups
control_clean = merged_clean[merged_clean["variant_number"] == 0]["active_mins"]
treatment_clean = merged_clean[merged_clean["variant_number"] == 1]["active_mins"]

# Re-plot Boxplot again after Outlier removal
plot_boxplot(control_clean, treatment_clean, "Control Group (Cleaned)", "Treatment Group (Cleaned)")

# Plot histogram and QQ plot again after Outlier removal
plot_histogram(control_clean, treatment_clean, "Control Group (Cleaned)", "Treatment Group (Cleaned)")
plot_qqplot(control_clean, treatment_clean, "Control Group (Cleaned)", "Treatment Group (Cleaned)")

# Compute Mean, Median, Standard Deviation and Variance again after Outlier removal
cleaned_stats = pd.DataFrame({
    "Control": [control_clean.mean(), control_clean.median(), control_clean.std(), control_clean.var()],
    "Treatment": [treatment_clean.mean(), treatment_clean.median(), treatment_clean.std(), treatment_clean.var()]
}, index=["Mean", "Median", "Std Dev", "Variance"])
print(cleaned_stats)

# Since data is NOT normal, we perform Mann-Whitney U-Test
u_test = stats.mannwhitneyu(control_clean, treatment_clean)
print("\nMann-Whitney U-Test Results:")
print(u_test)


################################################## PART 5 #########################################################

# Load the t3 dataset
t3 = pd.read_csv("Data/t3_user_active_min_pre.csv")

# Merge t3(pre-expeiment data) and t2 datasets
pre_merged = t3.merge(t2[['uid', 'variant_number']], on='uid', how='inner')

print("\nPre-Experiment Data:\n")
print(pre_merged.head())

# Remove outliers from pre-experiment data
pre_merged_clean = remove_outliers_iqr(pre_merged, "active_mins")

print(f"\nOutliers removed: {len(pre_merged) - len(pre_merged_clean)}")
print("Max Active Minutes After Outlier Removal:", merged_clean["active_mins"].max())

# Get total active minutes before and after the experiment (per user)
pre_experiment_agg = pre_merged_clean.groupby("uid").agg(pre_experiment_mins=("active_mins", "sum")).reset_index()
post_experiment_agg = merged_clean.groupby("uid").agg(post_experiment_mins=("active_mins", "sum")).reset_index()

# Find and keep only common users in both pre and post experiment datasets
common_users = set(pre_experiment_agg["uid"]).intersection(set(post_experiment_agg["uid"]))
engagement_data_filtered = pre_experiment_agg.merge(post_experiment_agg, on="uid", how="inner")
engagement_data_filtered = engagement_data_filtered.merge(t2[["uid", "variant_number"]], on="uid", how="inner") # Merged with user variant information

print("\nTotal Users in Pre-Experiment:", len(pre_experiment_agg))
print("Total Users in Post-Experiment:", len(post_experiment_agg))
print("Users in Both Periods:", len(engagement_data_filtered))

# Compare pre-experiment engagement between Control and Treatment groups
pre_experiment_stats = pre_experiment_agg.merge(t2[["uid", "variant_number"]], on="uid", how="inner")
pre_summary_stats = pre_experiment_stats.groupby("variant_number")["pre_experiment_mins"].describe()
print("\nSummary Statistics for Pre-Experiment Engagement:\n", pre_summary_stats)

# Perform Mann-Whitney U-Test on pre-experiment engagement
pre_control = pre_experiment_stats[pre_experiment_stats["variant_number"] == 0]["pre_experiment_mins"]
pre_treatment = pre_experiment_stats[pre_experiment_stats["variant_number"] == 1]["pre_experiment_mins"]

u_test_pre = stats.mannwhitneyu(pre_control, pre_treatment)
print("\nMann-Whitney U-Test Result (Pre-Experiment Engagement):", u_test_pre)

# Plot Histogram and Boxplot of Pre-Experiment Engagement
plot_histogram(pre_control, pre_treatment, label1="Pre-Experiment Control", label2="Pre-Experiment Treatment")
plot_boxplot(pre_control, pre_treatment, label1="Pre-Experiment Control", label2="Pre-Experiment Treatment")


# Calculate how much engagement changed for each user
engagement_data_filtered["engagement_gain"] = engagement_data_filtered["post_experiment_mins"] - engagement_data_filtered["pre_experiment_mins"]
print("\nEngagement Gain Data:\n", engagement_data_filtered.head())

# Compute summary statistics for engagement gain
summary_stats_gain = engagement_data_filtered.groupby("variant_number")["engagement_gain"].describe()
print("\nSummary Statistics for Engagement Gain:\n", summary_stats_gain)

# Separate engagement gain for Control and Treatment groups
control_gain = engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"]
treatment_gain = engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"]

# Perform Mann-Whitney U-Test
u_test_gain = stats.mannwhitneyu(control_gain, treatment_gain)
print("\nMann-Whitney U-Test Result (Engagement Gain):", u_test_gain)

# Plot Histogram and Boxplot of Engagement Gain for Control and Treatment Groups
plot_histogram(
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"],
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"],
    label1="Control Group Engagement Gain",
    label2="Treatment Group Engagement Gain"
)

plot_boxplot(
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"],
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"],
    label1="Control Group Engagement Gain",
    label2="Treatment Group Engagement Gain"
)


################################################## PART 6 #########################################################

# Load the t4 dataset
t4 = pd.read_csv("Data/t4_user_attributes.csv")

# Merge user attributes with engagement data
engagement_with_attributes = engagement_data_filtered.merge(t4, on="uid", how="inner")

# Compute summary statistics for engagement gain by gender and user type
gender_stats = engagement_with_attributes.groupby("gender")["engagement_gain"].agg(["mean", "median", "std"])
print("\nEngagement statistics by gender:\n", gender_stats)
user_type_stats = engagement_with_attributes.groupby("user_type")["engagement_gain"].agg(["mean", "median", "std"])
print("\nEngagement statistics by user type:\n", user_type_stats)

# Perform Mann-Whitney U-Test for engagement gain between male and female users
male_gain = engagement_with_attributes[engagement_with_attributes["gender"] == "male"]["engagement_gain"]
female_gain = engagement_with_attributes[engagement_with_attributes["gender"] == "female"]["engagement_gain"]
u_test_gender = stats.mannwhitneyu(male_gain, female_gain)
print("\nMann-Whitney U-Test Result (Male vs. Female Engagement Gain):", u_test_gender.pvalue)

# Create a boxplot to compare engagement gain by gender
plt.figure(figsize=(10, 6))
sns.boxplot(x=engagement_with_attributes["gender"], y=engagement_with_attributes["engagement_gain"], palette="Set2", hue=None)
plt.xlabel("Gender")
plt.ylabel("Engagement Gain")
plt.title("Engagement Gain by Gender")
plt.xticks(fontsize=12)
plt.show()

# Plot boxplot to compare engagement gain by user type
plt.figure(figsize=(10, 5))
sns.boxplot(data=engagement_with_attributes, x="user_type", y="engagement_gain", palette="coolwarm", hue=None)
plt.title("Engagement Gain by User Type")
plt.xlabel("User Type")
plt.ylabel("Engagement Gain")
plt.xticks(fontsize=12)
plt.show()