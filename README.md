# Israeli Election-Campaign SMS Classifier

A Hebrew NLP project that classifies SMS messages as `political` or
`non_political`. The notebook implements Multinomial Naive Bayes from scratch
and combines 10 independently trained, class-balanced learners by majority
vote.

## Highlights

- Hebrew-aware tokenization with optional stopword removal and bigrams
- Multinomial Naive Bayes implemented with NumPy and Python data structures
- Train-only 5-fold cross-validation and hyperparameter selection
- Balanced 10-model ensemble with majority voting
- Evaluation with political-class F1, confusion matrix, baseline comparison,
  error analysis, and indicative-token explainability
- Fixed Train/Test data loaded directly from the
  [Israeli Election SMS Filtering dataset](https://www.kaggle.com/datasets/yanivbahalul/israeli-election-sms-filtering)

## Run the notebook

1. Clone this repository.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start Jupyter and open
   `israeli_election_campaign_sms_classifier.ipynb`:

   ```bash
   jupyter notebook
   ```

The first notebook cell installs `kagglehub[pandas-datasets]` automatically for
Google Colab users. The dataset is downloaded from Kaggle when the notebook is
run, so the repository does not duplicate the dataset files.

## Method

The provided Train split is used for feature engineering, model selection, and
final training. The provided Test split remains untouched until final
evaluation. Each ensemble member trains on a different random class-balanced
sample, and the final prediction is chosen by majority vote.

## Project file

- `israeli_election_campaign_sms_classifier.ipynb` — complete analysis,
  implementation, experiments, results, and conclusions

