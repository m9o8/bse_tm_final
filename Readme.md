# Text Mining 2024-2025 - Final Project

Authors:

- Blanca Jimenez
- Maria Simakova
- Moritz Peist

## Analyzing How Economic Class is Represented in Literature Across Time Periods

This project analyzes how economic inequality is portrayed in fiction across different historical periods, testing whether literary representations correlate with actual economic conditions documented by economists like Thomas Piketty in "Capital in the Twenty-First Century."

## Project Overview

Literary works have long reflected societal conditions, including economic inequality. This project applies computational text analysis to examine how the portrayal of wealth, poverty, and class distinctions in literature has evolved in relation to actual economic conditions.

### Research Questions

- How does the language used to describe wealth and poverty in fiction reflect the economic inequality of the time period?
- Can we detect significant differences in how economic class is portrayed between periods of high inequality (Gilded Age/pre-1929) and lower inequality (post-WWII)?
- What linguistic features best predict a text's time period based on its economic content?

## Methodology

1. **Data Collection**: Curate a corpus of literary texts from high inequality and lower inequality periods, focusing on works with clear class distinctions.

2. **Text Analysis**:
   - Extract passages containing wealth/poverty descriptors
   - Perform sentiment analysis on economic references
   - Analyze linguistic features associated with different social classes

3. **Machine Learning**:
   - Train classifiers to predict a text's time period based on its economic language
   - Identify key features that differentiate high vs. low inequality periods

4. **Comparative Analysis**:
   - Compare literary representations with historical economic data
   - Test correlations between linguistic measures and inequality metrics

## Data Sources

- Public domain novels from Project Gutenberg
- Historical economic inequality metrics
- Works referenced by Piketty and other economic historians

## Technologies

- Python for text processing and analysis
- NLP libraries (NLTK, spaCy)
- Machine learning (scikit-learn)
- Data visualization libraries

## Expected Outcomes

- Insights into how literature reflects or responds to economic conditions
- Quantifiable measures of how class representation in fiction has evolved
- A classifier that can identify time periods based on economic language
- Visualizations of the relationship between literary and economic trends

## Team

This project is being developed as part of the Introduction to Text Mining and Natural Language Processing course.

## License

[MIT License](LICENSE)