from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have document-level TF-IDF matrices for pre and post periods
# pre_tfidf and post_tfidf are sparse matrices from TfidfVectorizer.transform()


def calculate_term_significance(pre_tfidf, post_tfidf, vocabulary):
    """
    Calculate statistical significance of TF-IDF differences between periods

    Args:
        pre_tfidf: Sparse matrix of TF-IDF values for pre-treatment documents
        post_tfidf: Sparse matrix of TF-IDF values for post-treatment documents
        vocabulary: List or dict of terms corresponding to matrix columns

    Returns:
        DataFrame with significance statistics for each term
    """
    results = []
    vocab_list = [term for term, idx in sorted(vocabulary.items(), key=lambda x: x[1])]

    # For each term (column in the TF-IDF matrices)
    for term_idx, term in enumerate(vocab_list):
        # Extract term's TF-IDF values for all documents in each period
        pre_values = pre_tfidf[:, term_idx].toarray().flatten()
        post_values = post_tfidf[:, term_idx].toarray().flatten()

        # Remove zeros to focus on documents where the term appears
        pre_values_nonzero = pre_values[pre_values > 0]
        post_values_nonzero = post_values[post_values > 0]

        # Calculate statistics
        pre_mean = pre_values.mean()
        post_mean = post_values.mean()
        pre_doc_freq = (pre_values > 0).sum() / len(pre_values)
        post_doc_freq = (post_values > 0).sum() / len(post_values)
        diff = post_mean - pre_mean

        # Perform statistical tests
        # 1. T-test on all values (including zeros)
        t_stat, p_value = stats.ttest_ind(post_values, pre_values, equal_var=False)

        # 2. T-test on non-zero values (when term is present)
        if len(pre_values_nonzero) > 0 and len(post_values_nonzero) > 0:
            nonzero_t, nonzero_p = stats.ttest_ind(
                post_values_nonzero, pre_values_nonzero, equal_var=False
            )
        else:
            nonzero_t, nonzero_p = np.nan, np.nan

        # 3. Chi-square test on document frequency
        pre_has_term = (pre_values > 0).sum()
        pre_no_term = len(pre_values) - pre_has_term
        post_has_term = (post_values > 0).sum()
        post_no_term = len(post_values) - post_has_term

        contingency = np.array(
            [[pre_has_term, pre_no_term], [post_has_term, post_no_term]]
        )

        try:
            chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        except ValueError:  # In case of issues with the contingency table
            chi2, chi2_p = np.nan, np.nan

        # 4. Calculate effect size (Cohen's d)
        # Pooled standard deviation
        pooled_std = np.sqrt(
            (
                (len(pre_values) - 1) * pre_values.var()
                + (len(post_values) - 1) * post_values.var()
            )
            / (len(pre_values) + len(post_values) - 2)
        )

        cohens_d = (post_mean - pre_mean) / pooled_std if pooled_std > 0 else np.nan

        # Store results
        results.append(
            {
                "term": term,
                "pre_mean_tfidf": pre_mean,
                "post_mean_tfidf": post_mean,
                "tfidf_diff": diff,
                "pre_doc_freq": pre_doc_freq,
                "post_doc_freq": post_doc_freq,
                "doc_freq_diff": post_doc_freq - pre_doc_freq,
                "t_statistic": t_stat,
                "p_value": p_value,
                "nonzero_t_stat": nonzero_t,
                "nonzero_p_value": nonzero_p,
                "chi2_statistic": chi2,
                "chi2_p_value": chi2_p,
                "cohens_d": cohens_d,
                # Flag for significance
                "is_significant": p_value < 0.05,
                "freq_significant": chi2_p < 0.05,
            }
        )

    # Convert to DataFrame
    return pl.DataFrame(results)


if __name__ == "__main__":
    # Step 1: Load and prepare your lemmatized text data
    # Assuming you have a dataframe with columns: 'post_id', 'date', 'lemmatized_text'
    # Example of loading data (replace with your actual data loading code)
    df = pl.scan_parquet(
        "../data/batched_processing/stackoverflow_processed_batch1.parquet"
    )

    # Step 2: Define pre and post-treatment periods
    treatment_date = datetime(2022, 11, 30)
    df = df.with_columns(
        pl.col("CreationDate")
        .cast(pl.Datetime)
        .lt(treatment_date)
        .alias("pre_treatment")
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
        max_features=5000,  # Consider top 5000 terms
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

    # Print results
    print("Top 20 terms with increased importance post-ChatGPT:")
    print(more_important.head(20))

    print("\nTop 20 terms with decreased importance post-ChatGPT:")
    print(less_important.head(20))

    print("\nTop 20 new or emerging terms post-ChatGPT:")
    print(new_terms.head(20))

    print("\nTop 20 disappearing terms post-ChatGPT:")
    print(disappeared_terms.head(20))

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
    # plt.show()

    ########################################### TF-IDF, further analysis ###################################################

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

    # Example usage with your existing TF-IDF data
    term_stats = calculate_term_significance(
        pre_tfidf,  # Your pre-treatment TF-IDF matrix
        post_tfidf,  # Your post-treatment TF-IDF matrix
        tfidf_vectorizer.vocabulary_,  # Your vocabulary dictionary
    )

    # Find statistically significant changes
    significant_terms = term_stats.filter(
        (pl.col("is_significant").cast(pl.Boolean))  # Significant TF-IDF difference
        & (
            pl.col("freq_significant").cast(pl.Boolean)
        )  # Significant doc frequency difference
        & (pl.col("cohens_d").abs() > 0.2)  # Meaningful effect size
    )

    # Sort by absolute effect size
    significant_terms = significant_terms.sort(
        pl.col("cohens_d").abs(), descending=True
    )

    # Print results
    print(f"Found {len(significant_terms)} statistically significant term changes:")
    print(
        significant_terms.select(
            ["term", "tfidf_diff", "p_value", "chi2_p_value", "cohens_d"]
        ).head(20)
    )

    # Visualize significant terms with their effect sizes
    plt.figure(figsize=(16, 8))
    top_terms = significant_terms.head(30)
    sns.barplot(data=top_terms.to_pandas(), x="cohens_d", y="term", palette="coolwarm")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.title("Terms with Statistically Significant Changes After ChatGPT")
    plt.xlabel("Effect Size (Cohen's d)")
    plt.tight_layout()
    plt.savefig("../imgs/significant_tfidf_terms.svg")

    # Add significance markers to your existing plot
    plt.figure(figsize=(16, 8))

    # Get significant terms
    sig_terms = set(significant_terms["term"].to_list())

    # Create visualization with the same terms as before
    top_increased = (
        vocab_table.filter(pl.col("tfidf_diff") > 0)
        .sort("tfidf_diff", descending=True)
        .head(15)
    )
    top_decreased = (
        vocab_table.filter(pl.col("tfidf_diff") < 0)
        .sort("tfidf_diff", descending=False)
        .head(15)
    )
    to_plot = pl.concat([top_increased, top_decreased])

    # Add significance indicator
    to_plot = to_plot.with_columns(
        pl.col("term").is_in(sig_terms).alias("is_significant")
    )

    # Convert to pandas for seaborn
    to_plot_pd = to_plot.to_pandas()

    # Create barplot
    ax = sns.barplot(
        data=to_plot_pd,
        x="tfidf_diff",
        y="term",
        hue="is_significant",  # Color by significance
        palette={
            True: "darkblue",
            False: "lightblue",
        },  # Different colors for significant terms
    )

    # Add a star to significant terms
    for idx, row in to_plot_pd.iterrows():
        if row["is_significant"]:
            ax.text(
                row["tfidf_diff"] * 1.05
                if row["tfidf_diff"] > 0
                else row["tfidf_diff"] * 0.95,
                idx,
                "*",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
            )

    plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    plt.title(
        "Change in TF-IDF Importance After ChatGPT (with Statistical Significance)"
    )
    plt.xlabel("Change in TF-IDF Score")
    plt.legend(["Not Significant", "Significant (p<0.05)"])
    plt.tight_layout()
    plt.savefig("../imgs/tfidf_significance_plot.svg")
    # plt.show()

    # Save the full vocabulary table
    vocab_table.write_parquet("../data/tfidf_vocabulary_comparison.parquet")