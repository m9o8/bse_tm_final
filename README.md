# Text Mining 2024-2025 - Final Project

## Authors

- Blanca Jimenez
- Maria Simakova
- Moritz Peist

## ChatGPT's Impact on Stack Overflow Question Patterns: An NLP Analysis

This project investigates how the introduction of Large Language Models (specifically ChatGPT in November 2022) has influenced the nature and patterns of questions on Stack Overflow, testing whether AI tools are creating a significant shift in how developers seek programming knowledge online.

## Project Overview

Stack Overflow has been the premier Q&A platform for programmers for over a decade, but ChatGPT represents a technological disruption in how programmers seek information. This project applies computational text analysis to examine how question content, complexity, and focus have changed before and after ChatGPT's release.

### Research Questions

- How has the introduction of ChatGPT changed the nature, complexity, and patterns of questions asked on Stack Overflow?
- Has there been a significant decrease in simple, common programming questions that AI can easily answer?
- Have Stack Overflow questions shifted toward more complex problems or specialized topics post-ChatGPT?
- What linguistic features best distinguish between pre-ChatGPT and post-ChatGPT Stack Overflow questions?

## Methodology

1. **Causal Analysis**:
   - Implement a difference-in-differences (DiD) approach comparing Stack Overflow to non-programming Stack Exchange sites
   - Use synthetic control methods to create a counterfactual "synthetic Stack Overflow"

2. **Text Analysis**:
   - Extract and analyze question complexity metrics
   - Perform topic modeling to identify shifting question focus areas
   - Analyze linguistic features and technical vocabulary usage

3. **Machine Learning**:
   - Train classifiers to distinguish pre-ChatGPT vs. post-ChatGPT questions
   - Identify key features that characterize the changing nature of questions
   - Develop complexity scoring metrics

4. **Comparative Analysis**:
   - Compare question patterns across programming domains
   - Test hypotheses about which question types have migrated to AI tools

## Data Sources

- Stack Overflow question data (January 2021 - March 2024)
- Control data from non-programming Stack Exchange sites
- Archive.org Stack Exchange data dumps

## Expected Outcomes

- Quantifiable evidence of how ChatGPT has changed developer knowledge-seeking behavior
- Insights into which types of programming questions humans still prefer to ask other humans
- A classifier that can identify pre vs. post-ChatGPT questions based on content features
- Visualizations showing the shifting landscape of online programming knowledge sharing

## Team

This project is being developed as part of the Introduction to Text Mining and Natural Language Processing course.

## License

[MIT License](LICENSE)

## Miscaleanous

- Command to convert Stata PDFs to SVGs: `inkscape --export-type="svg" sdid_nlp_trends102.pdf`
