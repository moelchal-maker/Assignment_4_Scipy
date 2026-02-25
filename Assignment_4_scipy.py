import os
import sys

import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# Locate CSV in common locations (no CLI args)
possible_paths = [
    'players_stats_by_season_full_details.csv',
    os.path.join(os.path.dirname(__file__), 'players_stats_by_season_full_details.csv'),
    os.path.expanduser('~/Downloads/players_stats_by_season_full_details.csv'),
    '/Users/mohamadel-chal/Downloads/players_stats_by_season_full_details.csv',
]
csv_path = None
for p in possible_paths:
    if p and os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print('Error: players_stats_by_season_full_details.csv not found')
    print('Place the CSV next to this script or in ~/Downloads/')
    sys.exit(1)

# Load the data
df = pd.read_csv(csv_path)

print("NBA PLAYER STATISTICS ANALYSIS")
print()

print("[1] FILTERING FOR NBA REGULAR SEASON DATA")
print("-" * 80)

# Filter dataset for NBA regular season only
nba_regular = df[(df['League'] == 'NBA') & (df['Stage'] == 'Regular_Season')].copy()
print(f"Total records in original dataset: {len(df)}")
print(f"Records after filtering for NBA Regular Season: {len(nba_regular)}")
print(f"Season range: {nba_regular['Season'].min()} to {nba_regular['Season'].max()}")

print("\n[2] FINDING PLAYER WITH MOST REGULAR SEASONS")
print("-" * 80)

# Count unique seasons per player
seasons_per_player = nba_regular.groupby('Player')['Season'].nunique().sort_values(ascending=False)
most_seasons_player = seasons_per_player.index[0]
num_seasons = seasons_per_player.iloc[0]

print(f"Player with most regular seasons: {most_seasons_player}")
print(f"Number of seasons played: {num_seasons}")
print(f"\nTop 10 players by seasons played:")
print(seasons_per_player.head(10))

print("\n[3] THREE POINT ACCURACY FOR EACH SEASON")
print("-" * 80)

# Get data for the player with most seasons
player_data = nba_regular[nba_regular['Player'] == most_seasons_player].copy()
player_data = player_data.sort_values('Season')

# Extract years from Season column (e.g., "1999 - 2000" -> 2000)
player_data['Year'] = player_data['Season'].str.split(' - ').str[1].astype(int)

# Calculate 3-point accuracy (3PM / 3PA)
player_data['3P_Accuracy'] = np.where(
    player_data['3PA'] > 0, 
    player_data['3PM'] / player_data['3PA'], 
    0
)

# Also calculate for reference
player_data['FG_Accuracy'] = np.where(
    player_data['FGA'] > 0,
    player_data['FGM'] / player_data['FGA'],
    0
)

print(f"\n{most_seasons_player}'s Three Point Statistics by Season:")
print(f"{'Season':<15} {'Year':<6} {'3PM':<6} {'3PA':<6} {'3P Acc':<10} {'FGM':<6} {'FGA':<6} {'FG Acc':<8}")
print("-" * 80)
for idx, row in player_data.iterrows():
    print(f"{row['Season']:<15} {int(row['Year']):<6} {int(row['3PM']):<6} {int(row['3PA']):<6} "
          f"{row['3P_Accuracy']:<10.4f} {int(row['FGM']):<6} {int(row['FGA']):<6} {row['FG_Accuracy']:<8.4f}")

print("\n[4] LINEAR REGRESSION FOR THREE POINT ACCURACY")
print("-" * 80)

# Prepare data for regression (only seasons with 3PA > 0)
regression_data = player_data[player_data['3PA'] > 0].copy()
X = regression_data['Year'].values
y = regression_data['3P_Accuracy'].values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(f"Linear Regression Results:")
print(f"  Slope: {slope:.6f}")
print(f"  Intercept: {intercept:.6f}")
print(f"  R-squared: {r_value**2:.6f}")
print(f"  P-value: {p_value:.6e}")
print(f"  Standard Error: {std_err:.6f}")

# Function for line of best fit
def fit_line(x):
    return slope * x + intercept

# Generate predictions for all seasons
all_years = regression_data['Year'].values
predictions = fit_line(all_years)

# Visualization code removed to avoid matplotlib dependency

print("\n[5] AVERAGE THREE POINT ACCURACY (INTEGRATION METHOD)")
print("-" * 80)

# Get the range of seasons played
min_year = regression_data['Year'].min()
max_year = regression_data['Year'].max()
year_range = max_year - min_year

print(f"Season range: {min_year} to {max_year}")
print(f"Year range difference: {year_range}")

# Integration using trapezoidal rule
# Integrate the line of best fit over the year range
years_for_integration = np.linspace(min_year, max_year, 100)
fitted_accuracies = fit_line(years_for_integration)
integral_result = np.trapezoid(fitted_accuracies, years_for_integration)
average_3p_accuracy_integrated = integral_result / year_range

# Calculate actual average
actual_average_3p = regression_data['3P_Accuracy'].mean()
actual_3pm_total = regression_data['3PM'].sum()
actual_3pa_total = regression_data['3PA'].sum()
actual_3p_accuracy = actual_3pm_total / actual_3pa_total if actual_3pa_total > 0 else 0

print(f"\nIntegrated Average 3P Accuracy: {average_3p_accuracy_integrated:.6f}")
print(f"Actual Average 3P Accuracy (by season mean): {actual_average_3p:.6f}")
print(f"Actual 3P Accuracy (total 3PM/total 3PA): {actual_3p_accuracy:.6f}")
print(f"\nDifference (Integrated - Actual by-season mean): {average_3p_accuracy_integrated - actual_average_3p:.6f}")
print(f"Difference (Integrated - Actual total): {average_3p_accuracy_integrated - actual_3p_accuracy:.6f}")
print(f"\nComparison: The integrated average is {abs(average_3p_accuracy_integrated - actual_3p_accuracy)*100:.2f}% {'higher' if average_3p_accuracy_integrated > actual_3p_accuracy else 'lower'} than the actual average.")

print("\n[6] INTERPOLATION FOR MISSING SEASONS (2002-2003 and 2015-2016)")
print("-" * 80)

# Get all seasons for this player
all_player_seasons = set(player_data['Season'].unique())
print(f"Seasons in dataset for {most_seasons_player}: {len(all_player_seasons)} seasons")

# Check for missing seasons
missing_2002_2003 = "2002 - 2003" not in all_player_seasons
missing_2015_2016 = "2015 - 2016" not in all_player_seasons

print(f"Missing 2002-2003 season: {missing_2002_2003}")
print(f"Missing 2015-2016 season: {missing_2015_2016}")

if missing_2002_2003 or missing_2015_2016:
    # Create interpolation function using available data
    years_with_data = regression_data['Year'].values
    accuracy_with_data = regression_data['3P_Accuracy'].values
    
    # Use scipy's interp1d for interpolation
    from scipy.interpolate import interp1d
    interp_func = interp1d(years_with_data, accuracy_with_data, kind='cubic', fill_value='extrapolate')
    
    # Estimate missing values
    if missing_2002_2003:
        estimated_2002_2003 = interp_func(2003)
        print(f"\nEstimated 3P Accuracy for 2002-2003: {estimated_2002_2003:.6f}")
    
    if missing_2015_2016:
        estimated_2015_2016 = interp_func(2016)
        print(f"Estimated 3P Accuracy for 2015-2016: {estimated_2015_2016:.6f}")
    
    # Visualization code removed to avoid matplotlib dependency

print("\n[7] STATISTICAL ANALYSIS FOR FGM AND FGA")
print("-" * 80)

# Calculate statistics for FGM and FGA
fgm_data = nba_regular['FGM'].dropna()
fga_data = nba_regular['FGA'].dropna()

fgm_mean = fgm_data.mean()
fgm_var = fgm_data.var()
fgm_skew = stats.skew(fgm_data)
fgm_kurtosis = stats.kurtosis(fgm_data)

fga_mean = fga_data.mean()
fga_var = fga_data.var()
fga_skew = stats.skew(fga_data)
fga_kurtosis = stats.kurtosis(fga_data)

print(f"\n{'Statistic':<20} {'FGM':<15} {'FGA':<15}")
print("-" * 50)
print(f"{'Mean':<20} {fgm_mean:<15.4f} {fga_mean:<15.4f}")
print(f"{'Variance':<20} {fgm_var:<15.4f} {fga_var:<15.4f}")
print(f"{'Skewness':<20} {fgm_skew:<15.4f} {fga_skew:<15.4f}")
print(f"{'Kurtosis':<20} {fgm_kurtosis:<15.4f} {fga_kurtosis:<15.4f}")

print(f"\nComparison between FGM and FGA:")
print(f"  Mean ratio (FGM/FGA): {fgm_mean/fga_mean:.4f}")
print(f"  Variance ratio (FGM/FGA): {fgm_var/fga_var:.4f}")
print(f"  Skewness difference: {abs(fgm_skew - fga_skew):.4f}")
print(f"  Kurtosis difference: {abs(fgm_kurtosis - fga_kurtosis):.4f}")

print(f"\nInterpretation:")
print(f"  - Both distributions are right-skewed (positive skew), indicating a tail towards higher values")
print(f"  - FGA has higher values overall (as expected, attempted > made)")
print(f"  - FGM skewness: {fgm_skew:.4f} - moderate positive skew")
print(f"  - FGA skewness: {fga_skew:.4f} - moderate positive skew")
print(f"  - FGM kurtosis: {fgm_kurtosis:.4f} - slightly {'heavy-tailed' if fgm_kurtosis > 3 else 'light-tailed'}")
print(f"  - FGA kurtosis: {fga_kurtosis:.4f} - slightly {'heavy-tailed' if fga_kurtosis > 3 else 'light-tailed'}")

# Visualization code removed to avoid matplotlib dependency

print("\n[8] T-TESTS ANALYSIS")
print("-" * 80)
# Ensure we're comparing the same number of observations for paired analysis
min_len = min(len(fgm_data), len(fga_data))
fgm_aligned = fgm_data.iloc[:min_len].values
fga_aligned = fga_data.iloc[:min_len].values

print(f"\nSample size for paired tests: {min_len}")

# Assumption checks: normality of the paired differences
diffs = fga_aligned - fgm_aligned
shapiro_p = None
if len(diffs) >= 3:
    shapiro_stat, shapiro_p = stats.shapiro(diffs)
    print(f"\nShapiro-Wilk test for normality of differences: statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6e}")
else:
    print("\nShapiro-Wilk test requires at least 3 paired observations; skipping normality test.")

# Choose paired parametric or nonparametric test based on normality
if shapiro_p is not None and shapiro_p > 0.05:
    print(f"\nPAIRED T-TEST (parametric)")
    t_stat_paired, p_val_paired = stats.ttest_rel(fgm_aligned, fga_aligned)
    print(f"  t-statistic: {t_stat_paired:.6f}")
    print(f"  p-value: {p_val_paired:.6e}")
    print(f"  Result: {'Significant' if p_val_paired < 0.05 else 'Not significant'} at α=0.05")
else:
    print(f"\nWILCOXON SIGNED-RANK TEST (nonparametric paired alternative)")
    try:
        w_stat, p_val_paired = stats.wilcoxon(fgm_aligned, fga_aligned)
        print(f"  statistic: {w_stat:.6f}")
        print(f"  p-value: {p_val_paired:.6e}")
        print(f"  Result: {'Significant' if p_val_paired < 0.05 else 'Not significant'} at α=0.05")
    except Exception as e:
        p_val_paired = None
        print(f"  Wilcoxon test failed: {e}")

# Effect size for paired data (Cohen's d for paired differences)
if len(diffs) > 1:
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    if sd_diff > 0:
        cohens_d = mean_diff / sd_diff
        print(f"\nCohen's d (paired): {abs(cohens_d):.4f}")
        print(f"Effect size interpretation: {'Negligible' if abs(cohens_d) < 0.2 else 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}")
    else:
        print("\nStandard deviation of differences is zero; cannot compute Cohen's d.")
else:
    print("\nNot enough paired observations to compute effect size.")

print("\nANALYSIS COMPLETE")
