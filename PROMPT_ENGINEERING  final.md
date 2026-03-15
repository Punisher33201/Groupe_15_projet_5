# 🧠 Prompt Engineering Documentation
## Project 5 – Pediatric Appendicitis Diagnosis with Explainable ML

**Task Selected:** Data Processing & Exploratory Data Analysis   
**AI Tool Used:** DeepSeek  
**Focus Area:** Outlier detection, missing value handling, correlation analysis, data normalization

---

## 📌 Overview

This document presents the prompt engineering process used during the **data preprocessing and exploratory data analysis** phase of the project. For each step, we provide:

- The **structured  prompt as written** following prompt engineering best practices
- The **result obtained** from DeepSeek
---

##  Prompt 1 – Debugging a Boxplot That Displays Nothing
Context: I am working on an EDA notebook for a medical dataset (pediatric appendicitis).
I am trying to display a boxplot for all numerical columns using matplotlib.

Problem: The following code runs without error but displays an empty plot with no boxes:

    columns = list(X.select_dtypes(include=['number']).columns)
    fig, ax = plt.subplots()
    ax.boxplot([X[col] for col in columns])
    ax.get_xticklabels(columns)
    plt.show()

Question: Why does the plot appear empty? Please identify the bug, explain it clearly,
and provide a corrected version of the code.

###Result Obtained
DeepSeek correctly identified that `ax.get_xticklabels(columns)` is a **getter**, not a setter — it retrieves existing labels instead of defining them. The fix was to replace it with `ax.set_xticklabels(columns)`.

```python
ax.set_xticklabels(columns)  # Corrected line
```

## Prompt 2 – Displaying All Boxplots in a 9×2 Subplot Grid

Context: I have a pandas DataFrame X with 17 numerical columns from a pediatric medical dataset.
I want to visualize the distribution of each numerical variable using boxplots.

Task: Generate Python code using matplotlib to:
1. Create a figure with subplots arranged in a 9-row × 2-column grid.
2. Display one boxplot per numerical column, with the column name as the subplot title.
3. Remove empty subplots if the number of columns is less than 9×2.
4. Handle missing values (NaN) before plotting.
5. Add the observation count (n=...) to each subplot.

Constraint: Use plt.subplots() and axes.flatten() for iteration.
### Result Obtained
DeepSeek generated a complete, well-structured solution including:
- Automatic subplot grid with `axes.flatten()`
- NaN removal per column with `.dropna()`
- Observation count annotation
- Optional enhanced version with mean/median lines and colored boxes
---

##  Prompt 3 – Fixing the Outlier Detection Logic

Context: I am implementing outlier detection using the IQR method on a medical dataset
with 782 observations and 17 numerical features.

Problem: My code returns values like {'Age': 776.0, 'BMI': 755.0, ...} which are
close to the total number of rows (782). This seems incorrect for outlier counts.

Here is my current code:
    outliers[column] = float(((Q1 - 1.5 * IQR < X[column]) | (X[column] > Q3 + 1.5 * IQR)).sum())

Question:
1. Why are the returned values so large (close to 782)?
2. What is wrong with the boolean condition?
3. Provide the corrected condition with an explanation of the logical error.
```

###  Result Obtained
DeepSeek explained that `Q1 - 1.5 * IQR < X[column]` is True for almost all values (it's the opposite of the intended check). The correct condition is:

```python
outliers[column] = ((X[column] < Q1 - 1.5 * IQR) | (X[column] > Q3 + 1.5 * IQR)).sum()
---

##  Prompt 4 – Choosing Between Mean and Median for Imputation

Context: I am preprocessing a pediatric medical dataset (appendicitis diagnosis).
The dataset contains numerical variables such as Length_of_Stay, CRP, WBC_Count, BMI.
Some of these variables have missing values and appear to have right-skewed distributions
with outliers (confirmed from boxplot analysis).

Question: For imputing missing values in this medical dataset:
1. Should I use the mean or the median? Justify based on the data characteristics.
2. Are there cases where one is clearly better than the other?
3. Provide a general rule and a code example using pandas fillna().
```

### Result Obtained
DeepSeek recommended the **median** for medical data because:
- It is robust to outliers (e.g., extreme `Length_of_Stay` values)
- Distributions in clinical datasets are typically right-skewed
- The mean is distorted by extreme cases (complications, long hospitalizations)

```python
X['Length_of_Stay'].fillna(X['Length_of_Stay'].median(), inplace=True)
---

##  Prompt 5 – Interpreting Correlation Results
Context: I generated a Pearson correlation heatmap between 17 numerical features
and the binary target variable (Appendicitis: complicated vs uncomplicated) in a
pediatric medical dataset.

Observation: The highest correlations I observe are:
- Length_of_Stay: ~0.66
- CRP: ~0.55
- WBC_Count: ~0.37
- Alvarado_Score: ~0.30

My interpretation: I believe there is no correlation because values are below 0.87.

Question:
1. Is my interpretation correct? What is the standard threshold for considering
   a correlation as "strong" in medical/clinical data?
2. Which features are most relevant for the predictive model based on these results?
3. What does a negative correlation with one target class imply for a binary variable?
```

###  Result Obtained
DeepSeek corrected the misunderstanding, explaining that in medical and behavioral sciences, `|r| ≥ 0.3` is already considered **moderate to strong**. The threshold of 0.87 is unrealistically high for real-world clinical data. Key findings:
- `Length_of_Stay (0.66)` → strong correlation → clinically logical
- `CRP (0.55)` → moderate-strong → important inflammatory marker
- The threshold scale (Cohen/Guilford) was clearly explained
---

##  Prompt 6 – Choosing and Applying RobustScaler
Context: I am preprocessing a pediatric medical dataset before training ML classifiers
(SVM, Random Forest, LightGBM). The dataset contains numerical features with confirmed
outliers (e.g., CRP, Length_of_Stay, WBC_Count) and right-skewed distributions.

Question:
1. What is RobustScaler and how does it work mathematically?
2. Why is it more appropriate than StandardScaler for this dataset?
3. When should RobustScaler NOT be used?
4. Provide a complete code example applying RobustScaler to X_train and X_test
   following the correct fit/transform pipeline (fit on train only).
```

###  Result Obtained
DeepSeek provided a thorough explanation including the mathematical formula, comparison table with StandardScaler, and correct usage:

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, not fit
```

Also clarified that tree-based models (Random Forest, LightGBM) don't require scaling.


## Prompt 7 – Correcting the Correlation Matrix Computation
Context: I am computing a correlation matrix between numerical features (X) and
one-hot encoded target variables (y0 = pd.get_dummies(y)) in a pandas DataFrame.

Error: The line `corr = X[columns].corr(y0)` raises an error because pandas .corr()
does not accept another DataFrame as argument.

Goal: Compute a matrix of shape (n_features × n_target_classes) containing the
Pearson correlation between each feature and each binary target column.

Question:
1. Explain why X.corr(y0) fails.
2. Provide 2 alternative approaches:
   a) Using a double for-loop to fill a DataFrame
   b) Using pandas corrwith() for a more concise solution
3. Show which approach is more efficient and why.
```

### Result Obtained
DeepSeek provided both approaches clearly:

```python
# Approach 1: Double loop
corr = pd.DataFrame(index=columns, columns=y0.columns)
for col_x in columns:
    for col_y in y0.columns:
        corr.loc[col_x, col_y] = X[col_x].corr(y0[col_y])

# Approach 2: corrwith (more concise)
corr = pd.DataFrame({col_y: X[columns].corrwith(y0[col_y]) for col_y in y0.columns})
```
---

## 📊 Effectiveness & Insights from Prompt Engineering

The structured prompts used throughout this project consistently produced accurate 
and complete answers. The key principles applied were:

| Principle | Impact |
|-----------|--------|
| **Always provide context** (dataset type, pipeline stage) | DeepSeek gave medically relevant advice instead of generic answers |
| **State the observed problem explicitly** | Bugs were identified faster and more precisely |
| **Break into numbered sub-questions** | Responses were structured and covered all aspects |
| **Mention data characteristics** (skewness, outliers) | Led to domain-appropriate recommendations |
| **Include the erroneous code or output** | DeepSeek pinpointed exact logical errors immediately |

### Key Insights Gained

- **Domain context matters:** Specifying "medical dataset with clinical outliers" 
  led to recommending **median** imputation and **RobustScaler** — both clinically justified.
- **Explicit hypotheses get corrected faster:** Stating the wrong assumption 
  (threshold of 0.87) in Prompt 5 allowed DeepSeek to immediately correct it 
  with the standard medical research scale (|r| ≥ 0.3).
- **Structured prompts reduce debugging time:** Including erroneous output 
  (outlier counts close to 782) gave DeepSeek enough context to identify 
  the inverted boolean condition instantly.

### Potential Improvements
- Include expected output format in every prompt.
- Specify library versions when relevant.
- Ask explicitly for edge cases handling.
---

*Documentation prepared as part of Coding Week 09–15 March 2026 – Project 5: Pediatric Appendicitis Diagnosis with Explainable ML*
