# %%
# Imports
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# %%
def test_term_significance(
    pre_tfidf, post_tfidf, feature_names, vocab, n_bootstrap=100, alpha=0.05
):
    """
    Test for statistical significance of changes in term frequencies

    Args:
        pre_tfidf: List of pre-ChatGPT frequencies
        post_docs: List of post-ChatGPT frequencies
        feature_names: List of feature names of the TF-IDF vectorizer
        vocab: List of terms to test
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        DataFrame with significance results
    """
    # Initialize TF-IDF vectorizer
    # vectorizer = TfidfVectorizer(vocabulary=vocab)
    #
    ## Get document-term matrices
    # pre_tfidf = vectorizer.fit_transform(pre_docs)
    # post_tfidf = vectorizer.transform(post_docs)

    # Get the term indices in the vocabulary
    term_indices = {
        term: idx
        for idx, term in enumerate(
            feature_names
        )  # enumerate(vectorizer.get_feature_names_out())
    }

    results = []

    for term in tqdm(vocab):
        if term not in term_indices:
            continue

        term_idx = term_indices[term]

        # Extract term TF-IDF values for each document
        pre_term_values = pre_tfidf[:, term_idx].toarray().flatten()
        post_term_values = post_tfidf[:, term_idx].toarray().flatten()

        # Observed difference in means
        observed_diff = post_term_values.mean() - pre_term_values.mean()

        # 1. Mann-Whitney U test
        u_stat, mw_pvalue = stats.mannwhitneyu(
            post_term_values, pre_term_values, alternative="two-sided"
        )

        # 2. Permutation test
        # Combine samples
        combined = np.concatenate([pre_term_values, post_term_values])
        n_pre = len(pre_term_values)
        n_post = len(post_term_values)

        # Perform permutation test
        perm_diffs = []
        for _ in range(1000):  # Fewer permutations for speed
            np.random.shuffle(combined)
            perm_pre = combined[:n_pre]
            perm_post = combined[n_pre:]
            perm_diff = perm_post.mean() - perm_pre.mean()
            perm_diffs.append(perm_diff)

        # Calculate p-value
        perm_pvalue = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

        # 3. Bootstrap confidence interval
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_pre = np.random.choice(
                pre_term_values, size=len(pre_term_values), replace=True
            )
            boot_post = np.random.choice(
                post_term_values, size=len(post_term_values), replace=True
            )

            boot_diff = boot_post.mean() - boot_pre.mean()
            bootstrap_diffs.append(boot_diff)

        # Calculate 95% confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        # Check if CI includes zero
        ci_significant = (ci_lower > 0 and ci_upper > 0) or (
            ci_lower < 0 and ci_upper < 0
        )

        # Store results
        results.append(
            {
                "term": term,
                "pre_mean": pre_term_values.mean(),
                "post_mean": post_term_values.mean(),
                "difference": observed_diff,
                "percent_change": observed_diff / pre_term_values.mean() * 100
                if pre_term_values.mean() > 0
                else np.nan,
                "mw_pvalue": mw_pvalue,
                "perm_pvalue": perm_pvalue,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant_05": (mw_pvalue < alpha)
                and (perm_pvalue < alpha)
                and ci_significant,
            }
        )

    return pl.DataFrame(results)


# %%
# Step 1: Load and prepare your lemmatized text data
# Assuming you have a dataframe with columns: 'post_id', 'date', 'lemmatized_text'
# Example of loading data (replace with your actual data loading code)
df = pl.scan_parquet("../data/batched_processing/stackoverflow_processed_batch.parquet")

# Step 2: Define pre and post-treatment periods
treatment_date = datetime(2022, 11, 30)
df = df.with_columns(
    pl.col("CreationDate").cast(pl.Datetime).lt(treatment_date).alias("pre_treatment")
)

# Step 3: Group texts by pre/post treatment
pre_texts = (
    df.filter((pl.col("pre_treatment")) & (pl.col("processed_text").is_not_null()))
    .select("processed_text")
    .collect()
)
post_texts = (
    df.filter(
        (pl.col("pre_treatment").not_()) & (pl.col("processed_text").is_not_null())
    )
    .select("processed_text")
    .collect()
)

# Convert to lists for scikit-learn
pre_texts_list = pre_texts["processed_text"].to_list()
post_texts_list = post_texts["processed_text"].to_list()

# Step 4: Create TF-IDF matrices for both periods
# You can adjust parameters based on your needs
tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,  # Consider top 5000 terms
    min_df=10,  # Ignore terms that appear in less than 10 documents
    max_df=0.7,  # Ignore terms that appear in more than 70% of documents
    ngram_range=(1, 2),  # Include unigrams and bigrams
)

# Step 5: Fit on all documents to ensure consistent vocabulary
all_texts = pre_texts_list + post_texts_list
tfidf_vectorizer.fit(all_texts)

# Step 6: Transform both pre and post text sets
pre_tfidf = tfidf_vectorizer.transform(pre_texts_list)
post_tfidf = tfidf_vectorizer.transform(post_texts_list)

# %%
# Step 7: Extract vocabulary and calculate mean TF-IDF scores for each period
vocabulary = tfidf_vectorizer.get_feature_names_out()
pre_mean_tfidf = np.asarray(pre_tfidf.mean(axis=0)).flatten()
post_mean_tfidf = np.asarray(post_tfidf.mean(axis=0)).flatten()

# Step 8: Create a vocabulary table with comparative statistics
vocab_table = pl.DataFrame(
    {
        "term": vocabulary,
        "pre_mean_tfidf": pre_mean_tfidf,
        "post_mean_tfidf": post_mean_tfidf,
    }
)

# Calculate differences and ratios
vocab_table = vocab_table.with_columns(
    [
        (pl.col("post_mean_tfidf") - pl.col("pre_mean_tfidf")).alias("tfidf_diff"),
        (pl.col("post_mean_tfidf") / pl.col("pre_mean_tfidf")).alias("tfidf_ratio"),
    ]
)

# Replace infinity values in ratio with a large number
vocab_table = vocab_table.with_columns(
    pl.when(pl.col("tfidf_ratio").is_infinite())
    .then(1000.0)  # Large value for terms not in pre-treatment
    .otherwise(pl.col("tfidf_ratio"))
    .alias("tfidf_ratio")
)

# %%
# Step 9: Calculate term frequencies (document occurrence)
pre_doc_count = len(pre_texts_list)
post_doc_count = len(post_texts_list)

pre_term_counts = (pre_tfidf > 0).sum(axis=0).A1  # Count docs containing term
post_term_counts = (post_tfidf > 0).sum(axis=0).A1

# Add frequency info to the vocabulary table
vocab_table = vocab_table.with_columns(
    [
        (pl.lit(pre_term_counts) / pre_doc_count).alias("pre_doc_freq"),
        (pl.lit(post_term_counts) / post_doc_count).alias("post_doc_freq"),
        (
            (pl.lit(post_term_counts) / post_doc_count)
            - (pl.lit(pre_term_counts) / pre_doc_count)
        ).alias("doc_freq_diff"),
    ]
)

# Step 10: Find the most distinctive terms in each period
# Terms that became more important after treatment
more_important = vocab_table.filter(pl.col("tfidf_diff") > 0).sort(
    "tfidf_diff", descending=True
)
# Terms that became less important after treatment
less_important = vocab_table.filter(pl.col("tfidf_diff") < 0).sort(
    "tfidf_diff", descending=False
)

# Step 11: Find new terms that weren't significant before
new_terms = (
    vocab_table.filter(pl.col("pre_doc_freq") < 0.01)
    .filter(pl.col("post_doc_freq") > 0.05)
    .sort("post_doc_freq", descending=True)
)
# Find terms that disappeared or significantly declined
disappeared_terms = (
    vocab_table.filter(pl.col("pre_doc_freq") > 0.05)
    .filter(pl.col("post_doc_freq") < 0.01)
    .sort("pre_doc_freq", descending=True)
)

# %%
# Print results
print("Top 20 terms with increased importance post-ChatGPT:")
print(more_important.head(20))

print("\nTop 20 terms with decreased importance post-ChatGPT:")
print(less_important.head(20))

print("\nTop 20 new or emerging terms post-ChatGPT:")
print(new_terms.head(20))

print("\nTop 20 disappearing terms post-ChatGPT:")
print(disappeared_terms.head(20))

# %%
# Step 12: Visualize the results
plt.figure(figsize=(12, 8))
# Plot top 15 terms with biggest absolute change
top_changed = (
    vocab_table.filter(
        (pl.col("pre_doc_freq") > 0.01) | (pl.col("post_doc_freq") > 0.01)
    )
    .sort("tfidf_diff", descending=True)
    .head(15)
)

bottom_changed = (
    vocab_table.filter(
        (pl.col("pre_doc_freq") > 0.01) | (pl.col("post_doc_freq") > 0.01)
    )
    .sort("tfidf_diff", descending=False)
    .head(15)
)

# Combine for visualization
to_plot = pl.concat([top_changed, bottom_changed])

# Convert to pandas for Seaborn
to_plot_pd = to_plot.to_pandas()

# Create a barplot of the differences
sns.barplot(data=to_plot_pd, x="tfidf_diff", y="term", palette="coolwarm")
plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
plt.title("Terms with Biggest Change in TF-IDF Importance After ChatGPT")
plt.xlabel("Change in TF-IDF Score (Post - Pre)")
plt.tight_layout()
plt.savefig("../imgs/tfidf_change_plot.svg")
plt.show()

# %% [markdown]
# ## Further analysis

# %%
# Calculate percentage changes (with handling for zero/near-zero values)
vocab_table = vocab_table.with_columns(
    [
        (
            pl.when(pl.col("pre_mean_tfidf") > 0.0001)
            .then(
                (pl.col("post_mean_tfidf") - pl.col("pre_mean_tfidf"))
                / pl.col("pre_mean_tfidf")
                * 100
            )
            .otherwise(None)
        ).alias("tfidf_percent_change")
    ]
)

# Filter to terms with sufficient presence
significant_terms = vocab_table.filter(
    (pl.col("pre_doc_freq") > 0.01) | (pl.col("post_doc_freq") > 0.01)
)

# Visualize percentage changes
plt.figure(figsize=(16, 8))
top_increased = (
    significant_terms.filter(pl.col("tfidf_diff") > 0)
    .sort("tfidf_percent_change", descending=True)
    .head(15)
)
top_decreased = (
    significant_terms.filter(pl.col("tfidf_diff") < 0)
    .sort("tfidf_percent_change")
    .head(15)
)
to_plot = pl.concat([top_increased, top_decreased])

# Create a barplot with percentage changes
ax = sns.barplot(
    data=to_plot.to_pandas(), x="tfidf_percent_change", y="term", palette="coolwarm"
)

# Add absolute values as text
for i, p in enumerate(ax.patches):
    if i < len(top_increased):
        term = top_increased["term"][i]
        abs_diff = top_increased["tfidf_diff"][i]
        pre_val = top_increased["pre_mean_tfidf"][i]
        post_val = top_increased["post_mean_tfidf"][i]
    else:
        j = i - len(top_increased)
        term = top_decreased["term"][j]
        abs_diff = top_decreased["tfidf_diff"][j]
        pre_val = top_decreased["pre_mean_tfidf"][j]
        post_val = top_decreased["post_mean_tfidf"][j]

    # Add text label with absolute values
    if p.get_width() > 0:
        ax.text(
            0,
            p.get_y() + p.get_height() / 2,
            f"{abs_diff:.5f} ({pre_val:.5f} → {post_val:.5f})",
            ha="left",
            va="center",
            fontsize=8,
        )
    else:
        ax.text(
            0,
            p.get_y() + p.get_height() / 2,
            f"{abs_diff:.5f} ({pre_val:.5f} → {post_val:.5f})",
            ha="right",
            va="center",
            fontsize=8,
        )

plt.title("Percentage Change in Term Importance After ChatGPT", fontsize=16)
plt.xlabel("Change in TF-IDF Score (%)", fontsize=14)
plt.ylabel("Term", fontsize=14)
plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
plt.tight_layout()
plt.savefig("../imgs/tfidf_percent_change_plot.svg")
plt.show()

# %%
print("Term frequency - statistical tests")

terms_to_test = top_changed["term"].to_list() + bottom_changed["term"].to_list()

term_stats = test_term_significance(
    pre_tfidf=pre_tfidf,
    post_tfidf=post_tfidf,
    feature_names=vocabulary,
    vocab=terms_to_test,
    n_bootstrap=100,
    alpha=0.05,
)

# Save to file
term_stats.write_parquet("../data/tfidf_term_significance.parquet")

# %%
t_stats = pl.read_parquet("../data/tfidf_term_significance.parquet")
t_stats.head()
