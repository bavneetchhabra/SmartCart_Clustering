# SmartCart Customer Segmentation

A customer segmentation project that uses unsupervised machine learning
to group customers according to demographic attributes, household
information, and purchasing behavior. The analysis is implemented in the
`SmartCart_Clustering.ipynb` Jupyter Notebook.

The notebook prepares customer data, engineers useful features, reduces
dimensionality with Principal Component Analysis (PCA), evaluates
possible cluster counts, and compares K-Means with Agglomerative
Clustering.

## Project Workflow

1.  Load the customer dataset.
2.  Inspect and handle missing values.
3.  Engineer customer age, tenure, total spending, and household-child
    features.
4.  Simplify education and marital-status categories.
5.  Remove unused/original columns and inspect outliers.
6.  Explore numeric relationships using a correlation heatmap.
7.  Encode categorical features and standardize the data.
8.  Apply PCA to project the standardized features into three
    dimensions.
9.  Explore the number of clusters using the Elbow Method and Silhouette
    Score.
10. Fit K-Means and Agglomerative Clustering models.
11. Summarize and visualize the resulting customer segments.

## Dataset

The notebook expects a CSV file named:

``` text
smartcart_customers.csv
```

Place the file in the same directory as `SmartCart_Clustering.ipynb`.

The notebook uses customer demographic, household, enrollment, purchase,
and campaign-response information. The fields explicitly referenced in
the notebook include:

-   `ID`
-   `Year_Birth`
-   `Education`
-   `Marital_Status`
-   `Income`
-   `Dt_Customer`
-   `Recency`
-   `Response`
-   `Kidhome`
-   `Teenhome`
-   `MntWines`
-   `MntFruits`
-   `MntMeatProducts`
-   `MntFishProducts`
-   `MntSweetProducts`
-   `MntGoldProds`

The CSV is not included with the notebook supplied for this analysis.
Dataset provenance, licensing, and exact row/column counts are not
specified in the notebook.

## Feature Engineering and Preprocessing

### Missing values

-   Missing values in `Income` are filled using the median income.
-   The notebook checks missing-value counts before and after
    imputation.

### Feature engineering

The notebook creates the following features:

-   `Age`: calculated as `2026 - Year_Birth`.
-   `Customer_Tenure_Days`: number of days between each customer's
    `Dt_Customer` date and the latest customer date in the dataset.
-   `Total_Spending`: sum of spending across wine, fruit, meat, fish,
    sweets, and gold products.
-   `Total_Children`: sum of `Kidhome` and `Teenhome`.

### Category grouping

-   Education categories are consolidated into `Undergraduate`,
    `Graduate`, and `Postgraduate`.
-   Marital-status categories are grouped into a new `Living_With`
    feature with `Partner` and `Alone` values.

### Column removal and outliers

The notebook removes the customer ID, original birth year, original
marital status, individual child-count columns, customer date, and
individual product-spending columns after creating the derived features.

It then filters out records where:

-   `Income` is 600,000 or higher.
-   `Age` is 90 or higher.

A pair plot is used to inspect selected features before filtering.

### Encoding and scaling

-   `Education` and `Living_With` are one-hot encoded using
    scikit-learn's `OneHotEncoder`.
-   The encoded feature matrix is standardized using `StandardScaler`.

## Dimensionality Reduction: PCA

Principal Component Analysis is applied to the standardized features
with `n_components=3`. The resulting three principal components are used
for 3D visualization and clustering.

The notebook displays the explained variance ratio for the three
components and plots customers in a three-dimensional PCA space.

## Choosing the Number of Clusters

Two approaches are explored:

-   **Elbow Method:** K-Means inertia (WCSS) is calculated for cluster
    counts from 1 to 10. `KneeLocator` is used to estimate an elbow.
-   **Silhouette Score:** silhouette scores are calculated for cluster
    counts from 2 to 10.

The notebook plots the results separately and together to help inspect
candidate cluster counts. The final clustering cells use four clusters.

## Clustering Algorithms

### K-Means

K-Means is fitted to the three PCA components with `n_clusters=4` and
`random_state=42`. The resulting labels are visualized in 3D.

### Agglomerative Clustering

Agglomerative Clustering is fitted to the same PCA representation with:

-   `n_clusters=4`
-   `linkage="ward"`

The notebook's written conclusion states that **Agglomerative Clustering
performs better**. No quantitative comparison metric between the two
final models is recorded, so this conclusion should be validated with
additional evaluation if the project is extended.

## Cluster Characterization

The notebook uses the Agglomerative Clustering labels to:

-   Count the number of customers in each cluster.
-   Visualize total spending against income, colored by cluster.
-   Calculate the mean of the available features for each cluster.

These summaries can help interpret differences in customer income and
purchasing behavior.

## Tech Stack

-   Python
-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   kneed
-   Jupyter Notebook

## Setup and Usage

### 1. Clone the repository

``` bash
git clone https://github.com/bavneetchhabra/SmartCart_Clustering.git
cd SmartCart_Clustering
```

Replace the placeholders with your GitHub repository URL and local
repository folder name.

### 2. Create and activate a virtual environment (recommended)

``` bash
python -m venv .venv
```

On macOS/Linux:

``` bash
source .venv/bin/activate
```

On Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter
```

### 4. Add the dataset

Place `smartcart_customers.csv` in the project directory, alongside
`SmartCart_Clustering.ipynb`.

### 5. Run the notebook

``` bash
jupyter notebook SmartCart_Clustering.ipynb
```

Run the notebook cells in order.

## Suggested Repository Structure

``` text
SmartCart-Customer-Segmentation/
├── SmartCart_Clustering.ipynb
├── smartcart_customers.csv
├── README.md
└── requirements.txt
```

Only upload the dataset if its license and sharing terms allow
redistribution. If you do not want to publish the CSV, add
`smartcart_customers.csv` to `.gitignore` and explain in this README how
users can obtain it.

## Notes and Potential Improvements

-   The notebook calculates an elbow estimate (`optimal_k`) but the
    final K-Means and Agglomerative models are both configured with four
    clusters. Confirm that four is the intended cluster count based on
    the plots and evaluation.
-   The K-Means loop in the Elbow Method does not set a `random_state`,
    so its inertia values may vary between runs.
-   The age feature uses the fixed year 2026. For reproducibility across
    years, make the reference year explicit as a project parameter or
    derive it from a documented analysis date.
-   The scaler and one-hot encoder are fitted to the full dataset, which
    is appropriate for this exploratory unsupervised workflow but should
    be considered carefully if the project is later adapted to predict
    outcomes for unseen customers.
-   The notebook's claim that Agglomerative Clustering performs better
    is not supported by a recorded quantitative comparison. Consider
    comparing silhouette scores for both final label sets and reviewing
    cluster sizes and interpretability.
-   Add a source and license for the dataset, plus saved plots or a
    short interpretation of each cluster, to make the project easier for
    others to reproduce and understand.

## License

No project license is specified in the notebook. Add a `LICENSE` file if
you want to define how others may use, modify, and distribute this
project.
