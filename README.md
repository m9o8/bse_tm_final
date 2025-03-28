# Overflow Under-Flowed: ChatGPT's Impact on Stack Overflow

This project investigates how ChatGPT's release has transformed question patterns on Stack Overflow, combining causal inference with text mining to measure both quantitative and qualitative impacts.

## Authors

- Blanca Jimenez
- Maria Simakova
- Moritz Peist

## Project Overview

The advent of Large Language Models (LLMs) has triggered a paradigm shift in how individuals seek and obtain technical information. Stack Overflow, as the premier programming question-and-answer platform, has long been the go-to resource for developers facing coding challenges. However, with the public release of ChatGPT in November 2022, developers gained access to an AI assistant capable of providing immediate, contextual programming guidance—potentially disrupting established knowledge-seeking patterns on specialized forums.

![Parallel Trends: Weekly indexed question counts](imgs/indexed_trends.svg)

Our dataset spans January 2021 to March 2024, with a focus on scripting languages (JavaScript, Python, R, and PHP) that represent the largest volume of Stack Overflow questions and are areas where early versions of ChatGPT demonstrated particular strength. For the causal analysis component, we incorporate data from four non-programming Stack Exchange forums as control units.

## Key Findings

- **Significant Volume Reduction**: ChatGPT caused a 39.5% reduction in scripting language questions (JavaScript, Python, R, and PHP)
- **Complexity Increase**: Questions have become significantly more complex after ChatGPT's introduction
- **Content Shifts**: Term importance analysis shows troubleshooting and technical terms increased, while basic programming concepts decreased

## Abstract

Applying the Technology Acceptance Model framework, we analyze how ChatGPT's perceived usefulness and ease of use have reshaped developers' information-seeking behavior. Using a Synthetic Difference-in-Differences approach with data spanning January 2021 to March 2024, we establish that ChatGPT caused a significant 39.5% reduction in scripting language questions (JavaScript, Python, R, and PHP). Beyond this volumetric decline, we demonstrate a statistically significant increase in question complexity following ChatGPT's introduction.

Our TF-IDF analysis reveals meaningful linguistic shifts: terms related to troubleshooting and technical infrastructure increased in importance, while basic programming concepts declined significantly. These findings align with recent research suggesting developers strategically allocate questions between platforms based on perceived usefulness for specific query types. Our research provides empirical evidence of how large language models reshape knowledge-sharing dynamics in technical communities, pointing to a complementary relationship between AI tools and human-moderated forums.

![Changes in Term Importance](imgs/term_significance_plot.svg)

## Research Questions & Approach

1. To what extent has ChatGPT's introduction causally affected question volume on Stack Overflow?
2. How has the nature and complexity of questions changed post-ChatGPT?

We approach these questions through a two-stage methodology:

1. **Establish causality** through a Synthetic Difference-in-Differences (SDID) framework, quantifying the volumetric impact while controlling for temporal trends
2. **Apply NLP analysis** to understand changes in term frequencies and track question complexity changes before and after ChatGPT's release

## Methodology

### Causal Impact Analysis

- Synthetic DiD approach to construct counterfactual for Stack Overflow
- Control groups: Mathematics, Physics, Superuser, and AskUbuntu forums
- Event study analysis to track effects over time

### Text Mining & Complexity Analysis

- Composite complexity score based on title length, body length, tag count, and code/technical expression length
- TF-IDF analysis with statistical significance testing
- Bootstrap confidence intervals for term importance changes

![Question Complexity Over Time](imgs/stackoverflow_vs_rest.svg)

## Complexity Score Analysis

We constructed a parsimonious complexity score for forum posts composed of 4 key elements:

1. Title length
2. Body length
3. Number of tags
4. Length of technical expressions (code blocks for programming forums, equations for Mathematics/Physics)

The standardized complexity score is calculated as:

```
Complexity Score = 1/4 * (Z(TagCount) + Z(TechExprLength) + Z(BodyLength) + Z(TitleLength))
```

Where Z represents the z-standardization of each metric across all questions in our dataset.

Our synthetic DiD analysis reveals a statistically significant increase in question complexity (0.059 standard deviations) following ChatGPT's release. This effect grew stronger over time, with the most recent period showing the largest impact (0.092 standard deviations), suggesting a fundamental shift in how developers utilize Stack Overflow rather than a temporary adjustment.

## Repository Structure and Processing Pipeline

Below is a visualization of our entire processing pipeline, showing how data flows through the different stages of analysis:

```mermaid
---
config:
  theme: mc
  look: neo
  layout: elk
---
flowchart TD
 subgraph extraction["1.Data Extraction  - 1_data_extraction.py"]
        so["Stack Overflow Data"]
        extract["process_stack_data()"]
        math["Mathematics Data"]
        physics["Physics Data"]
        superuser["SuperUser Data"]
        askubuntu["AskUbuntu Data"]
        so_all["All SO Questions (stackoverflow.parquet, 2.5 GB)"]
        so_script["Script Languages Only (stackoverflow_script.parquet, 0.8 GB)"]
  end
 subgraph preparation["2.Data Preparation - 2_eda.py"]
        prepare["prepare_forum_data()"]
        weekly["Weekly Aggregated Data"]
        transform["transform_for_parallel()"]
        parallel_data["Transformed Data for Parallel Trends"]
        stata_script["Script Questions .dta (so_all.dta)"]
        stata_combined["All Questions .dta (so_script.dta)"]
  end
 subgraph so_preprocess["3.1a Stack Overflow - preprocessing_batch_so.py"]
        extract_so["Extract Text & Code"]
        batch_so["Batch Processing"]
        preprocess_so["Text Preprocessing"]
        merge_so["Merge Batches"]
        so_processed["SO Processed Data"]
  end
 subgraph other_preprocess["3.1b Other Forums - preprocessing_other.py"]
        combine_others["Combine Non-SO Forums"]
        extract_other["Extract Text & Tech Expressions"]
        batch_other["Batch Processing"]
        preprocess_other["Text Preprocessing"]
        merge_other["Merge Batches"]
        other_processed["Other Forums Processed Data"]
  end
 subgraph preprocessing["3.Preprocessing"]
        so_preprocess
        other_preprocess
  end
 subgraph text_analysis["3.2-3.3 Text Analysis"]
        metrics["Calculate Text Metrics\n3_2_text_metrics.py"]
        complexity["Complexity Score"]
        nlp_data["NLP Metrics .dta (nlp.dta)"]
        tfidf["TF-IDF Analysis\n3_3_processing.py"]
        term_freq["Term Frequency Analysis"]
        term_significance["Statistical Term Significance"]
  end
 subgraph volume["4.1 Volume Analysis - 4_1_stata.do"]
        did_volume["DiD Analysis"]
        synthdid_volume["Synthetic DiD"]
        event_volume["Event Study"]
  end
 subgraph complexity_analysis["4.2 Complexity Analysis - 4_2_nlp.do"]
        did_nlp["DiD Analysis"]
        synthdid_nlp["Synthetic DiD"]
        event_nlp["Event Study"]
  end
 subgraph stats["4.Statistical Analysis in Stata"]
        volume
        complexity_analysis
  end
    source["Stack Exchange 7z Archives (100 GB)"] --> extraction
    extract --> so & math & physics & superuser & askubuntu
    so --> so_all & so_script
    so_all --> prepare
    so_script --> prepare & extract_so
    math --> prepare & combine_others
    physics --> prepare & combine_others
    superuser --> prepare & combine_others
    askubuntu --> prepare & combine_others
    prepare --> weekly
    weekly --> transform
    transform --> parallel_data
    parallel_data --> stata_script & stata_combined
    extract_so --> batch_so
    batch_so --> preprocess_so
    preprocess_so --> merge_so
    merge_so --> so_processed
    combine_others --> extract_other
    extract_other --> batch_other
    batch_other --> preprocess_other
    preprocess_other --> merge_other
    merge_other --> other_processed
    so_processed --> metrics & tfidf
    other_processed --> metrics
    metrics --> complexity
    complexity --> nlp_data
    tfidf --> term_freq
    term_freq --> term_significance
    stata_script --> did_volume
    stata_combined --> did_volume
    did_volume --> synthdid_volume
    synthdid_volume --> event_volume
    nlp_data --> did_nlp
    did_nlp --> synthdid_nlp
    synthdid_nlp --> event_nlp
    term_significance --> terms_results["Term Change Results"]
    event_volume --> volume_results["Volume Impact Results"]
    event_nlp --> complexity_results["Complexity Impact Results"]
    terms_results --> final["Final ChatGPT Impact Analysis"]
    volume_results --> final
    complexity_results --> final
     extract:::process
     so:::data
     math:::data
     physics:::data
     superuser:::data
     askubuntu:::data
     so_all:::data
     so_script:::data
     prepare:::process
     weekly:::data
     transform:::process
     parallel_data:::data
     stata_script:::data
     stata_combined:::data
     extract_so:::process
     preprocess_so:::process
     so_processed:::data
     extract_other:::process
     preprocess_other:::process
     other_processed:::data
     metrics:::process
     complexity:::data
     nlp_data:::data
     tfidf:::process
     term_freq:::data
     did_volume:::process
     synthdid_volume:::process
     did_nlp:::process
     synthdid_nlp:::process
     terms_results:::result
     volume_results:::result
     complexity_results:::result
     final:::result
    classDef process fill:#f9f,stroke:#333,stroke-width:2px
    classDef data fill:#bbf,stroke:#333,stroke-width:1px
    classDef result fill:#bfb,stroke:#333,stroke-width:2px
```

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

## Data Sources

- [Stack Overflow data (January 2021 - March 2024)](https://archive.org/download/stackexchange)
- [Control data from four non-programming Stack Exchange sites](https://archive.org/download/stackexchange)
- Focus on scripting languages (JavaScript, Python, R, PHP)

## Interpretation & Conclusions

Our findings support the hypothesis that ChatGPT has altered information-seeking behavior in programming communities. Developers now appear to reserve simpler questions for ChatGPT while turning to Stack Overflow for more complex programming challenges that require human expertise.

The empirical evidence points to a complementary relationship between AI-powered assistants and human-moderated Q&A forums, with each platform serving distinct informational needs within the programming community. Stack Overflow appears to be evolving toward a repository for more complex programming questions, while more straightforward queries may be increasingly handled through interaction with large language models like ChatGPT.

## License

[MIT License](LICENSE)

---

*This project was developed as part of the Introduction to Text Mining and Natural Language Processing course at Barcelona School of Economics.*
