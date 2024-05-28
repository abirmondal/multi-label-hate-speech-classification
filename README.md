# multi-label-hate-speech-classification
This academic work explores the problem of detecting toxic comments in Bengali and Hindi social media text, which is unstructured and inflectional. Manual filtering is hard and inefficient, so we use deep learning models like transformers to extract features automatically. We compared various fine-tuned transformer models and tried to do a comparative study of the models.

We study the problem of detecting toxic comments in Bengali and Hindi social media text, which is unstructured and has misspelt vulgar words. We have compared almost **9** models out of which **6** of them are multi-lingual and **3** of them are language-specific transformer models.

## Dataset
We have merged two datasets and modified the columns to create a new dataset. The datasets are:
- [Multi Labeled Bengali Toxic Comments](https://www.kaggle.com/datasets/tanveerbelaliut/multi-labeled-bengali-toxic-comments)
- [Hate-Speech-in-Hindi](https://www.kaggle.com/datasets/harithapliyal/hate-speech-in-hindi)

The merged dataset is available on *Kaggle* as a dataset named [Modified-hate-speech-bengali-hindi](https://www.kaggle.com/datasets/abirmondal/modified-hate-speech-bengali-hindi). For simplicity we have kept the Bengali and Hindi datasets in different folders.

## Dataset Division
We have divided the datasets into three parts:
| Train Set | Test Set | Validation Set |
| :-: | :-: | :-: |
| 16889 (70.00%) | 4856 (20.00%) | 2417 (10.00%) |

We have used sklearn's `train_test_split` function to divide the Bengali dataset. The Hindi dataset was divided already, so we used that division only.

## Models
We have used the following [Hugging Face](https://huggingface.co/) transformers:
1. [Twitter/twhin-bert-base](https://huggingface.co/Twitter/twhin-bert-base)
2. [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased)
3. [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased)
4. [distilbert/distilbert-base-multilingual-cased](distilbert/distilbert-base-multilingual-cased)
5. [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
6. [google/muril-base-cased](https://huggingface.co/google/muril-base-cased)
7. [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) \*
8. [l3cube-pune/hindi-bert-scratch](https://huggingface.co/l3cube-pune/hindi-bert-scratch) \*
9. [flax-community/roberta-hindi](https://huggingface.co/flax-community/roberta-hindi) \*

Language-specific models are marked by *'\*'*.

## Results
Metrics for different transformer models for the Merged Dataset (Hindi + Bengali)
| Model | Accuracy | F1-Score | ROC AUC | Hamming Loss | Jaccard Score | Zero-One Loss |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **twhin-bert-base** | **74.856** | **80.976** | **0.877** | **0.081** | **0.681** | **0.251** |
| bert-base-multiling (u) | 72.288 | 78.969 | 0.859 | 0.088 | 0.652 | 0.277 |
| bert-base-multiling (c) | 72.288 | 78.969 | 0.859 | 0.088 | 0.652 | 0.277 |
| distilbert-base-multiling | 70.583 | 78.406 | 0.855 | 0.089 | 0.645 | 0.294 |
| xlm-roberta-base | 73.315 | 80.351 | 0.871 | 0.082 | 0.671 | 0.267 |
| muril-base-cased | 56.758 | 69.331 | 0.779 | 0.112 | 0.531 | 0.432 |

We have achieved almost the best accuracy using the **twhin-bert-base** model. Our dataset is multi-labelled, so we focused mainly on *f1-score* to find the best model.

> Language-specific results will be available in our report, which will be available very soon.

## Testing
The transformer testing code is available in the file [Testing Transformer Models.ipynb](/testing-codes/Testing%20Transformer%20Models.ipynb)
Just change the value of `model` in the `pipeline` function arguments.
The fine-tuned models are:
* [abirmondalind/bert-base-multilingual-uncased-hate-speech-ben-hin](https://huggingface.co/abirmondalind/bert-base-multilingual-uncased-hate-speech-ben-hin)
* [abirmondalind/bert-base-multilingual-cased-hate-speech-ben-hin](https://huggingface.co/abirmondalind/bert-base-multilingual-cased-hate-speech-ben-hin)
* [kingshukroy/distilbert-base-multilingual-cased-hate-speech-ben-hin](https://huggingface.co/kingshukroy/distilbert-base-multilingual-cased-hate-speech-ben-hin)
* [kingshukroy/xlm-roberta-base-hate-speech-ben-hin](https://huggingface.co/kingshukroy/xlm-roberta-base-hate-speech-ben-hin)
* [kingshukroy/twhin-bert-base-hate-speech-ben-hin](https://huggingface.co/kingshukroy/twhin-bert-base-hate-speech-ben-hin)
* [abirmondalind/muril-base-cased-hate-speech-ben-hin](https://huggingface.co/abirmondalind/muril-base-cased-hate-speech-ben-hin)
* [arnabmukhopadhyay/bangla-bert-base-hate-speech-ben](https://huggingface.co/arnabmukhopadhyay/bangla-bert-base-hate-speech-ben) \*
* [arnabmukhopadhyay/Hindi-bert-v2](https://huggingface.co/arnabmukhopadhyay/Hindi-bert-v2) \*
* [arnabmukhopadhyay/Roberta-hindi-hate-speech](https://huggingface.co/arnabmukhopadhyay/Roberta-hindi-hate-speech) \*

Language-specific models are marked by *'\*'*.

> ❗REMEMBER to change the code for language-specific models.
