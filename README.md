# Assignment_4_Scipy

This project is a terminal-based statistical analysis of NBA regular season player data using Python, Pandas, NumPy, and SciPy. The script loads a comprehensive player statistics dataset, filters it to include only NBA regular season records, and performs a structured sequence of analytical procedures to evaluate player performance trends and league-wide statistical behavior.



## Dataset Loading and Filtering

The script automatically searches for the file `players_stats_by_season_full_details.csv` in common directories. Once located, it loads the dataset and filters it so that only records meeting the following conditions remain:

* League is `"NBA"`
* Stage is `"Regular_Season"`

It then reports:

* Total records in the original dataset
* Total records after filtering
* The overall season range covered



## Identifying the Longest-Career Player

The program groups the filtered data by player and counts the number of unique seasons played. It determines:

* The player with the most regular seasons
* The number of seasons played
* The top players ranked by total seasons

This player becomes the focus of the detailed shooting analysis.



## Season-by-Season Shooting Analysis

For the selected player, the script:

* Extracts the ending year from each season string
* Computes three-point accuracy (3PM ÷ 3PA)
* Computes overall field goal accuracy (FGM ÷ FGA)

It prints a formatted table showing each season’s totals and calculated efficiencies.



## Linear Regression Trend Analysis

To evaluate performance over time, the script performs linear regression on three-point accuracy by year. It calculates:

* Slope
* Intercept
* R-squared value
* P-value
* Standard error

This determines whether the player’s three-point shooting shows a statistically significant upward or downward trend.



## Average Three-Point Accuracy via Integration

Instead of relying only on arithmetic means, the script estimates a career-average accuracy using numerical integration. It:

* Constructs a regression line
* Integrates it across the career span using the trapezoidal rule
* Divides by the total year range

The integrated average is compared with:

* The mean of seasonal accuracies
* The overall accuracy computed from total makes divided by total attempts

The differences between these methods are displayed and interpreted.



## Interpolation of Missing Seasons

If certain seasons (such as 2002–2003 or 2015–2016) are missing for the selected player, the script applies cubic interpolation to estimate the missing three-point accuracy values based on surrounding data.



## League-Wide Distribution Analysis (FGM vs FGA)

The script performs statistical analysis on field goals made (FGM) and field goals attempted (FGA) across the NBA dataset. It computes:

* Mean
* Variance
* Skewness
* Kurtosis

It compares the distributions and provides interpretation regarding skewness, tail behavior, and variability differences between attempts and makes.



## Inferential Statistical Testing

The final section performs hypothesis testing to compare FGM and FGA values in a paired framework.

The process includes:

* Testing normality of paired differences using the Shapiro-Wilk test
* Performing a paired t-test if normality holds
* Performing a Wilcoxon signed-rank test if normality is violated
* Calculating Cohen’s d to measure effect size

The script reports statistical significance at the 0.05 level and classifies the magnitude of the observed effect.



## Program Structure

The entire workflow:

* Runs in the terminal
* Requires no command-line arguments
* Generates no visualizations
* Automatically handles missing season data
* Prints detailed statistical interpretation at each stage

## AI Usage
AI was used in the readme.md file as I had previously failed at producing a good readme file. Therfore I used AI for this one to produce a formate that I could use in the future based on what was seen here. Visual Studio code automatic AI was also used to resolve issues such as bugs and overall sections of code with the automatic suggestions. 

The program concludes by indicating that the analysis is complete.
