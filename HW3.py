
# Author: Bini Chandra
# Date: 02/03/2025
# This python file contains the code written for HW3.

# Importing necessary libraries
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
t1 = pd.read_csv("Data/t1_user_active_min.csv")
t2 = pd.read_csv("Data/t2_user_variant.csv")

################################################## PART 2 #########################################################

# Merge datasets on 'uid' (ignoring 'dt' from t2_variant since it's always 2019-02-06)
merged_data = t1.merge(t2[['uid', 'variant_number']], on='uid', how='inner')

# Show first five rows of the merged dataset
print("\nMerged Dataset:\n")
print(merged_data.head())


################################################## PART 3 #########################################################

def plot_histogram(control, treatment, label1="Control Group", label2="Treatment Group"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(control, bins=30, kde=True, color="blue", alpha=0.6)
    plt.title(f"Histogram of Active Minutes ({label1})")
    plt.xlabel("Active Minutes")
    
    plt.subplot(1, 2, 2)
    sns.histplot(treatment, bins=30, kde=True, color="blue", alpha=0.6)
    plt.title(f"Histogram of Active Minutes ({label2})")
    plt.xlabel("Active Minutes")
    
    plt.show()

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

def plot_boxplot(control, treatment, label1="Control Group", label2="Treatment Group"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=control, color="blue")
    plt.title(f"Boxplot of Active Minutes ({label1})")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=treatment, color="red")
    plt.title(f"Boxplot of Active Minutes ({label2})")
    
    plt.show()


def remove_outliers(data, threshold=1440):
    return data[data["active_mins"] <= threshold]


# Plot Boxplot of Control and Treatment groups
plot_boxplot(control_group, treatment_group)

# Remove outliers above 1,440 minutes
merged_clean = remove_outliers(merged_data)

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

# Merge t3 and t2 datasets on 'uid' (ignoring 'dt' from t2_variant since it's always 2019-02-06)
pre_merged = t3.merge(t2[['uid', 'variant_number']], on='uid', how='inner')
# Show first five rows of the merged dataset
print("\nMerged Dataset:\n")
print(pre_merged.head())


# Remove outliers
pre_merged_clean = remove_outliers(pre_merged)

# Display how many outliers were removed
print(f"\nOutliers removed: {len(pre_merged) - len(pre_merged_clean)}")
print("Max Active Minutes After Outlier Removal:", merged_clean["active_mins"].max())


# Separate Control and Treatment groups
pre_control_clean = pre_merged_clean[pre_merged_clean["variant_number"] == 0]["active_mins"]
pre_treatment_clean = pre_merged_clean[pre_merged_clean["variant_number"] == 1]["active_mins"]

# Aggregate pre-experiment engagement per user
pre_experiment_agg = pre_merged_clean.groupby("uid")["active_mins"].sum().reset_index()
pre_experiment_agg.rename(columns={"active_mins": "pre_experiment_mins"}, inplace=True)

# Aggregate post-experiment engagement per user
post_experiment_agg = merged_clean.groupby("uid")["active_mins"].sum().reset_index()
post_experiment_agg.rename(columns={"active_mins": "post_experiment_mins"}, inplace=True)

# Find common users in both pre- and post-experiment datasets
common_users = set(pre_experiment_agg["uid"]).intersection(set(post_experiment_agg["uid"]))

# Keep only users that exist in both periods
engagement_data_filtered = pre_experiment_agg.merge(post_experiment_agg, on="uid", how="inner")

# Merge with user variant information
engagement_data_filtered = engagement_data_filtered.merge(t2[["uid", "variant_number"]], on="uid", how="inner")

# Display user counts
print("\nTotal Users in Pre-Experiment:", len(pre_experiment_agg))
print("Total Users in Post-Experiment:", len(post_experiment_agg))
print("Users in Both Periods:", len(engagement_data_filtered))


# Compare pre-experiment engagement between Control and Treatment groups
pre_experiment_stats = pre_experiment_agg.merge(t2[["uid", "variant_number"]], on="uid", how="inner")

# Compute Summary Statistics for Pre-Experiment Data
pre_summary_stats = pre_experiment_stats.groupby("variant_number")["pre_experiment_mins"].describe()

print("\nSummary Statistics for Pre-Experiment Engagement:\n", pre_summary_stats)

# Perform Mann-Whitney U-Test on pre-experiment engagement
pre_control = pre_experiment_stats[pre_experiment_stats["variant_number"] == 0]["pre_experiment_mins"]
pre_treatment = pre_experiment_stats[pre_experiment_stats["variant_number"] == 1]["pre_experiment_mins"]

u_test_pre = stats.mannwhitneyu(pre_control, pre_treatment)

print("\nMann-Whitney U-Test Result (Pre-Experiment Engagement):", u_test_pre)

# Plot Histogram of Pre-Experiment Engagement
plot_histogram(pre_control, pre_treatment, label1="Pre-Experiment Control", label2="Pre-Experiment Treatment")

# Plot Boxplot of Pre-Experiment Engagement
plot_boxplot(pre_control, pre_treatment, label1="Pre-Experiment Control", label2="Pre-Experiment Treatment")


# Compute Engagement Gain
engagement_data_filtered["engagement_gain"] = engagement_data_filtered["post_experiment_mins"] - engagement_data_filtered["pre_experiment_mins"]

# Display engagement data with gain
print("\nEngagement Gain Data:\n", engagement_data_filtered.head())

# Compute summary statistics for engagement gain
summary_stats_gain = engagement_data_filtered.groupby("variant_number")["engagement_gain"].describe()

# Print results
print("\nSummary Statistics for Engagement Gain:\n", summary_stats_gain)

# Separate engagement gain for Control and Treatment groups
control_gain = engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"]
treatment_gain = engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"]

# Perform Mann-Whitney U-Test
u_test_gain = stats.mannwhitneyu(control_gain, treatment_gain)

# Print results
print("\nMann-Whitney U-Test Result (Engagement Gain):", u_test_gain)

# Plot Histogram of Engagement Gain for Control and Treatment Groups
plot_histogram(
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"],
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"],
    label1="Control Group Engagement Gain",
    label2="Treatment Group Engagement Gain"
)

# Plot Boxplot of Engagement Gain
plot_boxplot(
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 0]["engagement_gain"],
    engagement_data_filtered[engagement_data_filtered["variant_number"] == 1]["engagement_gain"],
    label1="Control Group Engagement Gain",
    label2="Treatment Group Engagement Gain"
)