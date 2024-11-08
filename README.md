# Text Classification Project: BERT and Word Embeddings

This project implements a text classification model that categorizes text into five predefined classes: **Politics**, **Sport**, **Technology**, **Entertainment**, and **Business**. It utilizes two major text embedding techniques, **Word2Vec** and **GloVe**, as well as **BERT** for fine-tuned performance. The project features an interactive interface built with **Gradio**, allowing users to input text and predict its category.

---

## Project Overview

The project classifies text into distinct categories using various text embedding models and a machine learning classifier. Initially, the model is trained with **Word2Vec** and **GloVe** embeddings, but further optimization is achieved with **BERT** fine-tuning for superior accuracy. The classification labels include Politics, Sport, Technology, Entertainment, and Business, which are used for training and evaluation.

### Key Features:

- **Word2Vec** and **GloVe** embeddings for text vectorization.
- **BERT** fine-tuning for state-of-the-art text classification.
- Interactive **Gradio** interface for live predictions.
- **Evaluation metrics**: Accuracy, Precision, Recall, and F1 Score.

### Performance Results

- **Word2Vec Model**:

  - Accuracy: 0.8685
  - Precision: 0.8695
  - Recall: 0.8685
  - F1 Score: 0.8678

- **GloVe Model**:

  - Accuracy: 0.9577
  - Precision: 0.9587
  - Recall: 0.9577
  - F1 Score: 0.9580

- **BERT Model**:
  - Accuracy: 0.9812 (Fine-tuned on the dataset)

### Data Summary

The dataset consists of labeled text data categorized into five classes:

- **Politics** = 0
- **Sport** = 1
- **Technology** = 2
- **Entertainment** = 3
- **Business** = 4

Each category label corresponds to a class for easier model training and analysis.

---

## Project Structure

The project includes the following key components:

1. **Data Preprocessing**:
   - Text cleaning (removing mentions, URLs, special characters, etc.).
2. **Vectorization**:
   - **Word2Vec**: Custom-trained word embeddings.
   - **GloVe**: Pre-trained embeddings (`glove-wiki-gigaword-100`).
3. **Model Training and Evaluation**:
   - Logistic Regression using Word2Vec and GloVe embeddings.
   - Fine-tuning with **BERT** for higher accuracy.
4. **Model Saving and Loading**:
   - Models are saved with `joblib` for future use.
5. **Gradio Interface**:
   - A simple interface for text input and category prediction using either Word2Vec, GloVe, or BERT.

---

## Dependencies

To run the project, install the required packages by running:

```bash
pip install -r requirements.txt
```

### Required Libraries:

- `numpy`
- `pandas`
- `plotly`
- `seaborn`
- `matplotlib`
- `gensim`
- `sklearn`
- `transformers`
- `torch`
- `gradio`
- `joblib`
- `wordcloud`

---


## Project Features

- **Interactive Visualizations**:

  - Pie chart and bar chart for label distribution.
  - Average word count visualization per category.
  - Word clouds for each category.

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, and F1 Score.
- **Word Embedding Models**:
  - **Word2Vec**: Custom-trained on project data.
  - **GloVe**: Pre-trained on a large corpus (Wikipedia + Gigaword).
- **BERT Fine-Tuning**: Improved accuracy with a transformer-based model.

---

## Example Use

To classify a text, you can use the following code:

```python
test_text = ["Artificial intelligence is revolutionizing industries by automating repetitive tasks, boosting productivity, and providing predictive analytics through large-scale data processing. Today, AI is embedded in sectors as diverse as healthcare, where algorithms enable early disease detection, and finance, where it aids in market trend analysis and risk management. The recent launch of a new smartphone has marked a technological turning point with an advanced camera system that rivals professional-grade devices, featuring high-resolution sensors and AI-optimized low-light photography. These innovations are redefining mobile photography, expanding creative possibilities for users."
]
predicted_category = predict(test_text)
print(f"Predicted category: {predicted_category}")
```

For real-time classification, use the **Gradio** interface where users can input text, and the model will predict the category.

---

## Conclusion

This project demonstrates the power of combining traditional word embeddings (Word2Vec, GloVe) with advanced transformer models (BERT) to solve text classification tasks effectively. With its user-friendly interface, it is accessible for both technical and non-technical users.( NANITH 777)
