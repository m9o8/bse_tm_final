# Text Mining 2024-2025 - Final Project

## Authors

- Blanca Jimenez
- Maria Simakova
- Moritz Peist

## ChatGPT's Impact on Stack Overflow Question Patterns: An NLP Analysis

This project investigates how Large Language Models (specifically ChatGPT, released in November 2022) have influenced question patterns on Stack Overflow, implementing an integrated framework combining synthetic difference-in-differences (DiD) methodology with text mining techniques to establish causal relationships between ChatGPT's introduction and changes in programming knowledge-seeking behavior.

## Project Overview

Stack Overflow has been the premier Q&A platform for programmers for over a decade, but ChatGPT represents a technological disruption in how programmers seek information. Our project applies robust causal inference methods combined with computational text analysis to examine changes in question volume, complexity, and content following ChatGPT's release.

### Research Questions & Hypotheses

1. **H1**: Simple, common programming questions have decreased in frequency post-ChatGPT, particularly for scripting languages (JavaScript, Python, R, PHP)
2. **H2**: Question complexity and specificity have increased as users turn to Stack Overflow for more challenging problems
3. **H3**: The distribution of question topics has shifted toward more specialized and advanced concepts
4. **H4**: The linguistic features of questions have changed in measurable ways through TF-IDF analysis

## Methodology

Our approach integrates causal inference with text mining through a two-phase analytical framework:

### 1. Data Extraction & Preparation

- Extract data from Stack Exchange archives (January 2021 - March 2024)
- Process questions from Stack Overflow and control forums (Mathematics, Physics, SuperUser, AskUbuntu)
- Focus specifically on scripting language questions (JavaScript, Python, R, PHP)
- Aggregate data to weekly level for time-series analysis

### 2. Text Preprocessing & Feature Extraction

- Batch process large datasets of HTML content from questions
- Extract separate components: text, code blocks, and technical expressions
- Implement NLP pipeline with tokenization, stopword removal, and lemmatization
- Calculate standardized complexity metrics:
  - Tag count (normalized)
  - Code/technical expression length (normalized)
  - Question body length (normalized)
  - Title length (normalized)
  - Composite complexity score across all features

### 3. Causal Framework: Synthetic DiD Analysis

- Implement synthetic control methodology to create counterfactual Stack Overflow trajectories
- Control for temporal trends using non-programming Stack Exchange sites
- Test parallel trends assumptions through transformations (log, indexed)
- Apply rigorous event study approach to measure temporal effects
- Calculate treatment effects for both:
  - Question volume impact (quantity effect)
  - Question complexity impact (quality effect)

### 4. Text Mining & Significance Testing

- Implement TF-IDF analysis to identify changing term importance
- Statistical testing of term frequency shifts (Mann-Whitney, permutation tests)
- Bootstrap confidence intervals for term importance changes
- Visualize significant changes in programming vocabulary post-ChatGPT

## Data Sources

- Stack Overflow question data (January 2021 - March 2024)
- Control data from four non-programming Stack Exchange sites
- Focus on scripting languages (JavaScript, Python, R, PHP)
- Archive.org Stack Exchange data dumps (.7z XML files)

## Technical Implementation

- **Data Processing**: Custom XML extraction pipeline using Polars and lxml
- **Text Analysis**: NLTK and spaCy for preprocessing, scikit-learn for TF-IDF
- **Statistical Analysis**: Synthetic DiD implemented in Stata
- **Visualization**: Seaborn and Matplotlib for Python, integrated with Stata outputs
- **Batch Processing**: Memory-optimized pipeline for processing multi-million question datasets

## Key Findings

- Quantifiable volume effects: Significant decrease in question frequency post-ChatGPT
- Complexity shifts: Increase in question complexity metrics following ChatGPT's introduction
- Term importance: Statistically significant changes in programming terminology usage
- Evidence of topic shifts toward more specialized concepts
- Heterogeneous effects across programming languages

## Repository Structure

```
├── code/
│   ├── 1_data_extraction.py         # Extract data from Stack Exchange archives
│   ├── 2_eda.py                     # Data preparation and exploratory analysis
│   ├── 3_1_preprocessing_batch_so.py # Batch processing for Stack Overflow
│   ├── 3_1_preprocessing_other.py   # Batch processing for control forums
│   ├── 3_2_text_metrics.py          # Calculate complexity metrics
│   ├── 3_3_processing.py            # TF-IDF and term significance analysis
│   ├── 4_1_stata.do                 # Volume analysis with synthetic DiD
│   └── 4_2_nlp.do                   # Complexity analysis with synthetic DiD
├── data/                            # Data directory (not in repo due to size)
└── imgs/                            # Output visualizations
```

##

In the following a sketch of our entire processing pipeline (subject to change):

```mermaid
flowchart TD
    %% Data Sources
    source[Stack Exchange 7z Archives] --> extraction

    %% Main Process Flow
    subgraph extraction[1.Data Extraction  - 1_data_extraction.py]
        extract["process_stack_data()"] --> so[Stack Overflow Data]
        extract --> math[Mathematics Data]
        extract --> physics[Physics Data]
        extract --> superuser[SuperUser Data]
        extract --> askubuntu[AskUbuntu Data]
        
        %% Stack Overflow filtering
        so --> so_all["All SO Questions (stackoverflow.parquet)"]
        so --> so_script["Script Languages Only (stackoverflow_script.parquet)"]
    end

    %% Data Preparation
    subgraph preparation[2.Data Preparation - 2_eda.py]
        so_all & so_script & math & physics & superuser & askubuntu --> prepare["prepare_forum_data()"]
        prepare --> weekly[Weekly Aggregated Data]
        weekly --> transform["transform_for_parallel()"]
        transform --> parallel_data[Transformed Data for Parallel Trends]
        
        %% Data export for statistical analysis
        parallel_data --> stata_script["Script Questions .dta (so_all.dta)"]
        parallel_data --> stata_combined["All Questions .dta (so_script.dta)"]
    end

    %% Preprocessing
    subgraph preprocessing[3.Preprocessing]
        subgraph so_preprocess[3.1a Stack Overflow - preprocessing_batch_so.py]
            so_script --> extract_so[Extract Text & Code]
            extract_so --> batch_so[Batch Processing]
            batch_so --> preprocess_so[Text Preprocessing]
            preprocess_so --> merge_so[Merge Batches]
            merge_so --> so_processed[SO Processed Data]
        end
        
        subgraph other_preprocess[3.1b Other Forums - preprocessing_other.py]
            math & physics & superuser & askubuntu --> combine_others[Combine Non-SO Forums]
            combine_others --> extract_other[Extract Text & Tech Expressions]
            extract_other --> batch_other[Batch Processing]
            batch_other --> preprocess_other[Text Preprocessing]
            preprocess_other --> merge_other[Merge Batches]
            merge_other --> other_processed[Other Forums Processed Data]
        end
    end

    %% Text Analysis
    subgraph text_analysis[3.2-3.3 Text Analysis]
        so_processed --> metrics[Calculate Text Metrics\n3_2_text_metrics.py]
        other_processed --> metrics
        
        metrics --> complexity[Complexity Score]
        complexity --> nlp_data["NLP Metrics .dta (nlp.dta)"]
        
        so_processed --> tfidf[TF-IDF Analysis\n3_3_processing.py]
        tfidf --> term_freq[Term Frequency Analysis]
        term_freq --> term_significance[Statistical Term Significance]
    end

    %% Statistical Analysis
    subgraph stats[4.Statistical Analysis in Stata]
        subgraph volume[4.1 Volume Analysis - 4_1_stata.do]
            stata_script --> did_volume[DiD Analysis]
            stata_combined -->
            did_volume --> synthdid_volume[Synthetic DiD]
            synthdid_volume --> event_volume[Event Study]
        end
        
        subgraph complexity_analysis[4.2 Complexity Analysis - 4_2_nlp.do]
            nlp_data --> did_nlp[DiD Analysis]
            did_nlp --> synthdid_nlp[Synthetic DiD]
            synthdid_nlp --> event_nlp[Event Study]
        end
    end

    %% Results Flow
    term_significance --> terms_results[Term Change Results]
    event_volume --> volume_results[Volume Impact Results]
    event_nlp --> complexity_results[Complexity Impact Results]
    
    %% Final Integration
    terms_results & volume_results & complexity_results --> final[Final ChatGPT Impact Analysis]

    %% Style
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef data fill:#bbf,stroke:#333,stroke-width:1px;
    classDef result fill:#bfb,stroke:#333,stroke-width:2px;
    
    class extract,prepare,transform,extract_so,extract_other,preprocess_so,preprocess_other,metrics,tfidf,did_volume,did_nlp,synthdid_volume,synthdid_nlp process;
    class so,math,physics,superuser,askubuntu,so_all,so_script,weekly,parallel_data,stata_script,stata_combined,so_processed,other_processed,complexity,nlp_data,term_freq data;
    class terms_results,volume_results,complexity_results,final result;
```

## Team

This project is being developed as part of the Introduction to Text Mining and Natural Language Processing course.

## License

[MIT License](LICENSE)
