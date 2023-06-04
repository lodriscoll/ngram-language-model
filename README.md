## N-Gram Language Model

This code implements an N-gram language model and provides functionality for tokenization, dataset loading, and perplexity calculation. The main purpose of the code is to create and evaluate N-gram language models. The supported N-gram models include unigram, bigram, and trigram models. Smoothing techniques, such as additive smoothing, are available for these models. Linear Interpolation between the three models is also included in the code.

## Usage

1. Clone the repository:

```shell
git clone https://github.com/lodriscoll/ngram-language-model.git
```

2. Navigate to the project directory:

```shell
cd ngram-language-model
```

3. There is no need to install any dependencies --  this code was implemented using pythons standard library

4. Download the required dataset using the shell script provided

``` shell
./subsample_1b_benchmark.sh
```

5. Run the NgramLanguageModel script with desired arguments to train the models:

```shell
python NgramLanguageModel.py [-h] [--eval EVAL_FILE] [--alpha ALPHA] [--lambdas LAMBDA1 LAMBDA2 LAMBDA 3]
```
options:

  --eval EVAL_FILE      Path to the evaluation dataset file (default: 1b_benchmark.dev.tokens)

  --alpha ALPHA         alpha parameter for additive smoothing

  --lambdas LAMBDA1 LAMBDA2 LAMBDA3
                        lambda values for linear interpolation, must sum to one

6. View the perplexity results


### Code Description

Here is a brief description of each function:

- `load_and_tokenize_dataset(path_to_file)`: Loads the dataset from a file and tokenizes it into sentences. The function returns a list of tokenized sentences.

- `replace_oov_tokens(tokenized_dataset)`: Replaces out-of-vocabulary (OOV) tokens in the dataset with the `<UNK>` token. OOV tokens are identified based on a frequency threshold (default is 3).

- `build_ngram_model(tokenized_dataset, n)`: Builds an N-gram language model based on the tokenized dataset and the specified N. The function calculates the probability of each N-gram using maximum likelihood estimation (MLE) and returns a dictionary of N-gram probabilities.

- `build_ngram_model_with_smoothing(tokenized_dataset, n, alpha)`: Builds a smoothed N-gram language model with additive smoothing. The function calculates the probability of each N-gram with smoothing and returns a dictionary of smoothed N-gram probabilities.

- `calculate_perplexity(test_data, ngram_model)`: Calculates the perplexity of a test dataset using an N-gram language model. The function returns the perplexity value.

- `main(args)`: The main function that parses command-line arguments, loads the datasets, builds and evaluates the language models, and prints the perplexity results.

In the `main` function, you can see that the code loads the training, development, and test datasets using the `load_and_tokenize_dataset` function. It then builds the N-gram language model using the `build_ngram_model` or `build_ngram_model_with_smoothing` functions depending on the specified options. Finally, it evaluates the language model on the test dataset by calculating the perplexity using the `calculate_perplexity` function.