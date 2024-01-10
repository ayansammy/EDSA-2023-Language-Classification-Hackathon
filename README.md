# EDSA 2023 Classification Hackathon

## Overview
South Africa stands as a multicultural society, distinguished by its profound linguistic diversity. Language, being a vital instrument, not only serves to enhance democracy but also plays a pivotal role in shaping the multifaceted aspects of social, cultural, intellectual, economic, and political life within the South African society.

Marked by the assurance of equal status for each, the country embraces a multilingual landscape with 11 official languages. The majority of South Africans exhibit multilingualism, proficiently speaking two or more of these designated official languages. Given the widespread multilingual proficiency among the population, it becomes evident that our systems and devices must effectively communicate in a diverse linguistic environment.

# Objectives
In this intriguing challenge, I embarked on a journey to:
- Implement a language identification system utilizing advanced NLP techniques to analyze diverse texts.
- Attain precise language identification across South Africa's 11 official languages for effective communication.
- Enable algorithm to conduct language-specific processing, accommodating the country's linguistic diversity.
- Develop a proficient system recognizing and interpreting texts in all 11 official languages, enhancing communication.
- Create a versatile system supporting applications reliant on language-specific processing, boosting overall functionality.
- Tailor the language identification algorithm to seamlessly integrate with South Africa's linguistic nuances, ensuring inclusivity and accuracy.

# Tools and Technology
- Development Language: Python 🐍
- Machine Learning Libraries: Utilizing scikit-learn and nltk 📚
- Experiment Tracking: Employing Comet for systematic recording 📜
- Model Evaluation: Assessing model efficiency through Kaggle submission on an unseen dataset.

# Data Preprocessing
Here are some of the Data Preprocessing highlights:
A dedicated function was employed to eliminate **Noise** in the dataset by performing the following operations:
- Removal of punctuations.
- Conversion of text to lowercase.
- Elimination of links.
- Removal of extra spaces.
- Handling of other special characters.

**Additional NLP preprocessing techniques were applied, including text tokenization, removal of stop words, and stemming using SnowballStemmer.**

# Exploratory Data Analysis
The dataset includes both training and test set data, covering 11 distinct South African languages. The following visual provides an overview of the distribution of these 11 languages.
![Language Distribution](visuals/download.png)
The language distribution plot reveals a well-balanced representation, with approximately 30,000 instances for each of South Africa's 11 unique languages. Each language is visually distinguished by a unique color, such as green for Xhosa, blue for English, orange for Sepedi, red for Venda, and so forth. This color-coded approach enhances clarity, aiding in easy identification and differentiation of each language. The uniform distribution ensures a diverse dataset for robust training of a language identification system, while the distinct colors contribute to clear visual analysis.

## Analyzing Each Language Separately
**Visualizing Xhosa Language Texts**
![Xhosa](visuals/xhosa.png)
The WordCloud for Xhosa Language reveals the prominence of specific words, including "okanye," "ukuba," "kufuneka," and more. These boldened words represent distinctive linguistic elements commonly used in Xhosa. Their prevalence underscores the rich presence of Xhosa language in the dataset, emphasizing its significant contribution to the overall linguistic diversity of the text data.
