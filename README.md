# Text Classification Project

This project classifies text data into five distinct categories: **Politics, Sport, Technology, Entertainment, and Business**. It uses both Word2Vec and GloVe embeddings with Logistic Regression as the classifier. The project is implemented with a user-friendly interface built using Gradio.

---

## Project Overview

This project is an NLP-based classifier designed to predict the category of text data based on pre-defined classes. Using advanced text embedding techniques (Word2Vec and GloVe), the classifier transforms raw text into meaningful vector representations, which are then used to train and evaluate a Logistic Regression model.

### Data Summary

The dataset contains text samples categorized into five classes:

- **Politics** = 0
- **Sport** = 1
- **Technology** = 2
- **Entertainment** = 3
- **Business** = 4

Each category is represented as a label in the dataset for easier model training and analysis.

### Performance Results

The performance of each model is summarized below:

#### Word2Vec Model

- **Accuracy**: 0.8685
- **Precision**: 0.8695
- **Recall**: 0.8685
- **F1 Score**: 0.8678

#### GloVe Model

- **Accuracy**: 0.9577
- **Precision**: 0.9587
- **Recall**: 0.9577
- **F1 Score**: 0.9580

The GloVe model outperforms the Word2Vec model, making it a preferred choice for this task.

## Project Structure

The project consists of the following major components:

- **Data Preprocessing**: Text cleaning, removing mentions, URLs, special characters, and unnecessary whitespaces.
- **Vectorization**:
  - **Word2Vec**: Trained using the corpus to create word embeddings.
  - **GloVe**: Leveraged using pre-trained embeddings from `glove-wiki-gigaword-100`.
- **Model Training and Evaluation**: Logistic Regression model trained on both Word2Vec and GloVe embeddings.
- **Model Saving and Loading**: Models are saved using `joblib` and can be reloaded for use in Gradio.
- **Gradio Interface**: A user-friendly interface where users can input text and select the model (Word2Vec or GloVe) for predictions.

## Dependencies

The following packages are required to run the project:

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

Install them via:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Run the Project**:
   Launch the Gradio interface with:

   ```bash
   python model.ipynb
   ```

3. **Use the Interface**:
   - Enter your text in the input box.
   - Choose whether to use the GloVe model for predictions.

## Project Features

- **Interactive Visualization**:
  - Pie chart and bar chart for label distribution.
  - Average word count per category visualization.
  - Word clouds for each category.
- **Word Embedding Models**:
  - **Word2Vec**: Custom-trained on the project data.
  - **GloVe**: Pre-trained on a large corpus (Wikipedia + Gigaword).
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score.
