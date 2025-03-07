from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    # Step 1: Load and prepare your lemmatized text data
    # Assuming you have a dataframe with columns: 'post_id', 'date', 'lemmatized_text'
    # Example of loading data (replace with your actual data loading code)
    df = pl.scan_parquet(
        "../data/batched_processing/stackoverflow_processed_batch.parquet"
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
    plt.savefig("../imgs/tfidf_change_plot.png")
    plt.show()

    # Save the full vocabulary table
    vocab_table.write_parquet("../data/tfidf_vocabulary_comparison.parquet")
