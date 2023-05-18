# Emotions classification
__Disclaimer__

This is a toy project: 
- it was done in ~10 h and needs a lot of upgrades for acceptable quality.
- model was train on subset of 16k images due to limitations in time and GPU resources.

## Task description
Given: labeled [dataset](https://huggingface.co/datasets/dair-ai/emotion) of tweets in English with 6 classes (*_sadness, joy, love, anger, fear, surprise_*).
#### Steps:
1. EDA
2. choose metrics
3. train classification model
4. compress the model
5. make conclusions about the results and provide ideas for further improvement

## Implemented features
- Dataset analysis ([notebook](EmotionsClassification_EDA.ipynb))
- RNN (LSTM) training ([train script](train.py))
- Model evaluation (accuracy, F1-score/Precision/Recall by class) ([test script](test.py))
- Comparing to Transformers pretrained model ([script for transformer's testing](test_transformer.py), [model link](https://huggingface.co/Vasanth/bert-base-uncased-finetuned-emotion))
- Model compression to fp16
 
## Results
- Training was done on a set on 16k items.
- Results below were tested on a test set of 2k items.
- BERT model was taken from huggingface hub just for comparison (I didn't train it myself).

### About the model
LSTM Encoder was used:
- n_layers: 3
- embedding_size: 256
- hidden size: 64
- dropout rate: 0.2

Loss: Cross-Entropy

### Quality of classification
| class    |   Precision |   Recall |   F1-score |   Support |
|----------|-------------|----------|------------|-----------|
| sadness  |       0.951 |    0.967 |      0.959 |       581 |
| joy      |       0.934 |    0.954 |      0.944 |       695 |
| love     |       0.828 |    0.786 |      0.806 |       159 |
| anger    |       0.980 |    0.891 |      0.933 |       275 |
| fear     |       0.839 |    0.933 |      0.884 |       224 |
| surprise |       0.857 |    0.636 |      0.730 |        66 |

### Confusion matrix
|     |   sadness |   joy |   love |   anger |   fear |   surprise |
|----------|-----------|-------|--------|---------|--------|------------|
| **sadness**  |       562 |     6 |      2 |       1 |     10 |          0 |
| **joy**      |         5 |   663 |     23 |       0 |      4 |          0 |
| **love**     |         2 |    30 |    125 |       2 |      0 |          0 |
| **anger**    |        14 |     2 |      1 |     245 |     13 |          0 |
| **fear**     |         6 |     0 |      0 |       2 |    209 |          7 |
| **surprise** |         2 |     9 |      0 |       0 |     13 |         42 |

### Model quality comparison
As classes are unbalanced, I used macro averaging for F1-score, Precision and Recall.
| Model       | Accuracy | F1-score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| BERT        | 0.9210   | 0.8786   |   0.8812  | 0.8808 |
| LSTM (fp32) | 0.9230   | 0.8761   |   0.8982  | 0.8613 |
| LSTM (fp16) | 0.9230   | 0.8761   |   0.8982  | 0.8613 |

### Model speed
Time in ms per one batch of size 32 (GPU: GTX 1050ti).
| Model       |    GPU   |   CPU    | 
|-------------|----------|----------|
| BERT        | 457.0829 | 1662.61  |       
| LSTM (fp32) |  3.5891  | 18.9171  |       
| LSTM (fp16) |  3.4194  |    -     |    

### Model size
| Model       |   Size   |   
|-------------|----------|
| BERT        |  438 MB  |     
| LSTM (fp32) |   9 MB   |        
| LSTM (fp16) |  4.5 MB  |   

## Analysis
### About the data
- Classes are unbalanced 
- Size of tweets is small (<100 characters), thus we can use Reccurent models
- Size of tweets doesn't vary much between different classes
- Texts are not separable using vanilla Bag-of-Words embeddings, though we can see some tiny clusters (groups of points of the same class) on embeddings visualization with t-SNE

### About the metrics
As classes are unbalanced, we should use metrics that handle this issue. 
During training the best checkpoint was chosen by F1-score.
On evaluation stage we use many metrics to catch different aspects and problems of the model:
- F1-score, Precision, Recall
- Confusion matrix
- Accuracy

### About the results
- The LSTM model has a good quality for a baseline solution.
- It seems to me quite strange that BERT model has the same quality. Probably the BERT model,  that I compared my model to, was poorly trained, and with better training it is possible to get higher results. That's why one of the first next steps that I propose is to retrain Transformer model.
- After model compression to fp16 quality stayed the same and model size decreased twice.

### Further ideas
1. Use full dataset
2. Weighted training or resampling
2. Add visualization of training process
3. Research training hyperparameters
2. Train transformer on full dataset to check the quality of pretrained (3rd party) model
3. Do Distillation with transformer model
4. Compress size and speed with quantization/pruning
