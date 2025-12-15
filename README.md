## Overview
Movie box office performance is an important topic in both industry and academic research, informing investment decisions, marketing strategy, and our understanding of audience behaviour. This study examines how well simple movie attributes can explain box office outcomes without relying on complex models. Using a dataset of 6,897 films released between 2006 and 2015, we fit linear regression and mixture-of-regressions models to explain box office performance using only genre, distributor, MPAA rating, release time, and a small set of title-based indicators. Our best model achieves an $R^2$ of 0.68, suggesting that basic metadata captures much of the variation in the data but is not sufficient for accurate explanation. Richer information such as budget, marketing, franchise status, and word-of-mouth effects is likely essential for more accurate analysis.

## File Structure

The repo is structured as:
- data/movies contains the raw data as obtained from https://dasl.datadescription.com/datafile/movies_06-15/.
- data/cleaned_movies contains the cleaned dataset that was constructed.
- model contains the three models fitted in this analysis, including linear regression, mixture model regression and a lasso regression.Outputs from each models and Diagnostics plots are also included.
- paper contains the files used to generate the paper, including the Quarto document, the PDF generated from Quarto, and the bibliography file.
- scripts contains the Python scripts used to load and clean the raw data. Tests for the data were also included.

## Statement on LLM usage
During this analysis, ChatGPT-5 was used to help debug code and suggest improvements to code structure. All code was subsequently reviewed and revised by the author. ChatGPT-5 was also used to assist with plot formatting in the Quarto document. In addition, ChatGPT-5 was used to polish wording for grammar and clarity; all ideas and substantive content are the author’s own.
