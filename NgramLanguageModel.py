import math
from collections import Counter
import argparse

def main():
    # add command line arguments to make experimentation seamless
    parser = argparse.ArgumentParser(description="Tokenization and Dataset Loading")
    parser.add_argument("--eval", metavar="EVAL_FILE", type=str, default='1b_benchmark.dev.tokens', help="Path to the evaluation dataset file (default: %(default)s)")
    parser.add_argument("--alpha", metavar="ALPHA", type=int, default=1, help="alpha parameter for additive smoothing")
    parser.add_argument('--lambdas', nargs='+', type=int, default=(0.1, 0.3, 0.6), help='lambda values for linear interpolation, must sum to one')
    args = parser.parse_args()


    # load and tokenize train and eval set
    raw_train_set, eval_set= load_and_tokenize_dataset('1b_benchmark.train.tokens'), load_and_tokenize_dataset(args.eval)

    # pre process the raw dataset by removing tokens with counts less than 3
    train_set = replace_oov_tokens(raw_train_set)

    # Build unigram model
    print("training unigram model...")
    unigram_model = build_unigram_model(train_set)

    # Build bigram model
    print("training bigram model...")
    bigram_model = build_bigram_model(train_set)

    # Build trigram model
    print("training trigram model...")
    trigram_model = build_trigram_model(train_set)

    # Build unigram model
    print("training unigram model with smoothing...")
    unigram_model_smoothed = build_unigram_model_with_smoothing(train_set, alpha=args.alpha)

    # Build bigram model
    print("training bigram model with smoothing...")
    bigram_model_smoothed = build_bigram_model_with_smoothing(train_set, alpha=args.alpha)

    # Build trigram model
    print("training trigram model with smoothing...")
    trigram_model_smoothed = build_trigram_model_with_smoothing(train_set, alpha=args.alpha)

    # Evaluate the models
    print("calculating unigram perplexity...")
    unigram_perplexity = calculate_unigram_perplexity(eval_set, unigram_model)

    print("calculating bigram perplexity...")
    bigram_perplexity = calculate_bigram_perplexity(eval_set, bigram_model)

    print("calculating trigram perplexity...")
    trigram_perplexity = calculate_trigram_perplexity(eval_set, trigram_model, bigram_model)

    print("calculating unigram with smoothing perplexity...")
    unigram_smoothed_perplexity = calculate_unigram_perplexity(eval_set, unigram_model_smoothed)

    print("calculating bigram with smoothing perplexity...")
    bigram_smoothed_perplexity = calculate_bigram_perplexity(eval_set, bigram_model_smoothed)

    print("calculating trigram with smoothing perplexity...")
    trigram_smoothed_perplexity = calculate_trigram_perplexity(eval_set, trigram_model_smoothed, bigram_model_smoothed)

    print("calculating linear interpolation smoothing perplexity...")
    linear_perplexity = calculate_linear_perplexity(eval_set, unigram_model,trigram_model, bigram_model, lambdas=args.lambdas)

    # Print the perplexities
    print(f"Unigram Perplexity: {unigram_perplexity}")
    print(f"Bigram Perplexity: {bigram_perplexity}")
    print(f"Trigram Perplexity: {trigram_perplexity}")
    print(f"Smoothed Unigram Perplexity: {unigram_smoothed_perplexity}")
    print(f"Smoothed Bigram Perplexity: {bigram_smoothed_perplexity}")
    print(f"Smoothed Trigram Perplexity: {trigram_smoothed_perplexity}")
    print(f"Linear Perplexity: {linear_perplexity}")


def load_and_tokenize_dataset(path_to_file):
    # load data into list of lists by first splitting on newlines
    # then concatenating special tokens on to list of tokens 
    # produced by splitting on spaces
    with open(path_to_file, 'r', encoding='utf-8') as file:
        raw_dataset = file.read()
        tokenized_dataset = [['<START>'] + s.split(' ') + ['<END>'] for s in raw_dataset.split('\n')]
    return tokenized_dataset

def flatten_list(tokenized_dataset):
    return [token for sentence in tokenized_dataset for token in sentence]

def replace_oov_tokens(tokenized_dataset):
    # create dictionary of num occurences
    token_counts = Counter(flatten_list(tokenized_dataset))

    # keep token if token_count is greater than 
    # or equal to three else <UNK>
    processed_dataset = [['<UNK>' if token_counts[token] < 3 else token for token in sentence] for sentence in tokenized_dataset]

    return processed_dataset

def build_unigram_model(tokenized_dataset):
    # get number of occurences of each unique unigram and length of all tokens
    token_counts = Counter(flatten_list(tokenized_dataset))
    total_tokens = sum(count for word, count in token_counts.items() if word != '<START>')
    
    # calculate unigram probabilities using MLE and return
    unigram_probs = {token: count / total_tokens for token, count in token_counts.items()}
    return unigram_probs

def build_bigram_model(tokenized_dataset):
    # get count of bigrams and unigrams
    bigram_counts = Counter((sentence[i], sentence[i+1]) for sentence in tokenized_dataset for i in range(len(sentence) - 1))
    unigram_counts = Counter(flatten_list(tokenized_dataset))
    
    # calculate bigram probabilites uisng MLE
    bigram_probs = dict()
    for (first_word, second_word), num_of_bigram in bigram_counts.items():
        bigram_probs[(first_word, second_word)] = num_of_bigram / unigram_counts[first_word]

    return bigram_probs

def build_trigram_model(tokenized_dataset):
    # get trigram and bigram counts
    trigram_counts = Counter((sentence[i], sentence[i+1], sentence[i+2]) for sentence in tokenized_dataset for i in range(len(sentence) - 2))
    bigram_counts = Counter((sentence[i], sentence[i+1]) for sentence in tokenized_dataset for i in range(len(sentence) - 1))

    # calculate trigram probabilities using MLE
    trigram_probs = dict()
    for (first_word, second_word, third_word), num_of_tris in trigram_counts.items():
        trigram_probs[(first_word, second_word, third_word)] = num_of_tris / bigram_counts[(first_word, second_word)]

    return trigram_probs

def build_unigram_model_with_smoothing(tokenized_dataset, alpha=1):
    # get number of occurences of each unique unigram and length of all tokens
    token_counts = Counter(flatten_list(tokenized_dataset))
    total_tokens = sum(count for word, count in token_counts.items() if word != '<START>')
    # vocab is set of all words minus <start> token
    V = len(token_counts) - 1
    
    # calculate unigram probabilities using MLE and return
    unigram_probs = {token: (count + alpha) / (total_tokens + V*alpha) for token, count in token_counts.items()}
    return unigram_probs

def build_bigram_model_with_smoothing(tokenized_dataset, alpha=1):
    # get count of bigrams and unigrams
    bigram_counts = Counter((sentence[i], sentence[i+1]) for sentence in tokenized_dataset for i in range(len(sentence) - 1))
    unigram_counts = Counter(flatten_list(tokenized_dataset))
    # vocab is set of all words minus <start> token
    V = len(set(flatten_list(tokenized_dataset))) - 1

    # calculate bigram probabilites uisng MLE
    bigram_probs = dict()
    for (first_word, second_word), num_of_bigram in bigram_counts.items():
        bigram_probs[(first_word, second_word)] = (num_of_bigram + alpha) / (unigram_counts[first_word] + V*alpha)

    return bigram_probs

def build_trigram_model_with_smoothing(tokenized_dataset, alpha=1):
    # get trigram and bigram counts
    trigram_counts = Counter((sentence[i], sentence[i+1], sentence[i+2]) for sentence in tokenized_dataset for i in range(len(sentence) - 2))
    bigram_counts = Counter((sentence[i], sentence[i+1]) for sentence in tokenized_dataset for i in range(len(sentence) - 1))
    # vocab is set of all words minus <start> token
    V = len(set(flatten_list(tokenized_dataset))) - 1

    # calculate trigram probabilities using MLE
    trigram_probs = dict()
    for (first_word, second_word, third_word), num_of_tris in trigram_counts.items():
        trigram_probs[(first_word, second_word, third_word)] = (num_of_tris + alpha) / (bigram_counts[(first_word, second_word)] + V*alpha)

    return trigram_probs

def calculate_unigram_perplexity(test_data, unigram_model):
    # init summation vars
    total_log_probability = 0
    total_words = 0
    unk = unigram_model.get('<UNK>')
    
    # calculate total log prob
    for sentence in test_data:
        for word in sentence[1:]:
            probability = unigram_model.get(word, 0)
            log_probability = math.log2(probability if probability > 0 else unk)  
            
            total_log_probability += log_probability
            
            total_words += 1
        
    # calculate perplexity
    avg_log_probability = total_log_probability / total_words
    perplexity = 2 ** (-avg_log_probability)
    
    return perplexity

def calculate_bigram_perplexity(test_data, bigram_model):
    # init summation vars
    total_log_probability = 0
    total_words = 0
    unk = 0.039
    
    # calculate total log prob
    for sentence in test_data:
        for i in range(len(sentence) - 1):
            current_word = sentence[i]
            next_word = sentence[i+1]
            
            bigram = (current_word, next_word)
            
            probability = bigram_model.get(bigram, 0)
            log_probability = math.log2(probability) if probability > 0 else unk
            
            total_log_probability += log_probability
            
            total_words += 1
    
    # calculate perplexity
    avg_log_probability = total_log_probability / total_words
    perplexity = 2 ** (-avg_log_probability)
    
    return perplexity

def calculate_trigram_perplexity(test_data, trigram_model, bigram_model):
    # init sumattion vars
    total_log_probability = 0
    total_words = 0
    unk = 0.039
    
    # calculate total log prob
    for sentence in test_data:
        for i in range(len(sentence) - 2):
            first_word = sentence[i]
            second_word = sentence[i+1]
            third_word = sentence[i+2]
            
            trigram = (first_word, second_word, third_word)
            if first_word == '<START>':
                probability = bigram_model.get((first_word, second_word), 0)
            else :
                probability = trigram_model.get(trigram, 0)
            log_probability = math.log2(probability) if probability > 0 else unk
            
            total_log_probability += log_probability
        
        total_words += len(sentence) - 1

    # calculate perplexity
    avg_log_probability = total_log_probability / total_words
    perplexity = 2 ** (-avg_log_probability)
    
    return perplexity

# function for part 3
def calculate_linear_perplexity(eval_set, unigram_model, trigram_model, bigram_model, lambdas):
    # init vars
    lambda1, lambda2, lambda3 = lambdas
    total_log_probability = 0
    total_words = 0
    unk = 0.00001


    for sentence in eval_set:
        # get bigram probs for first two words because trigrams start at index 2 
        if tuple(sentence[0:2]) in bigram_model:
            #check if first word after <START> is in unigram model
            if sentence[1] in unigram_model:
                #if it is we initialize the unigram and bigram perplexity values for first two words 
                total_log_probability += math.log2(lambda1 * unigram_model[sentence[1]] + lambda2 * bigram_model[tuple(sentence[0:2])] + lambda3 * bigram_model[tuple(sentence[0:2])])
            else:
                # else we initialize for just the bigram perplexity and trigram which is equal to bigram since we have only seen two words
                total_log_probability += math.log2(lambda1 * 0 + lambda2 * bigram_model[tuple(sentence[0:2])] + lambda3 * bigram_model[tuple(sentence[0:2])])     

        for i in range(len(sentence) - 2):
            # build ngrams
            trigram = (sentence[i], sentence[i+1], sentence[i+2])
            bigram = (sentence[i+1], sentence[i+2])
            unigram = sentence[i+2]
            
            # get probabilities
            trigram_probability = trigram_model.get(trigram, 0)
            bigram_probability = bigram_model.get(bigram, 0)
            unigram_probability = unigram_model.get(unigram, 0)
            
            # interpolate, calculate log probability, and add to running total
            interpolated_probability = lambda1*unigram_probability + lambda2*bigram_probability + lambda3*trigram_probability
            interpolated_log_probability = math.log2(interpolated_probability) if interpolated_probability > 0 else unk
            total_log_probability += interpolated_log_probability

        total_words+= len(sentence) - 1

    # calculate avg log prob and perp
    avg_log_probability = total_log_probability / total_words
    perplexity = 2 ** (-avg_log_probability)
    
    return perplexity


if __name__ == "__main__":
    main()



