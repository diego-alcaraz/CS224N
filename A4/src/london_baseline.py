# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    eval_corpus_path = "birth_dev.tsv"
    len_eval = len(open(eval_corpus_path, "r").readlines())
    predictions = ["London"] * len_eval

    #accuracy = utils.evaluate_places("dev.txt", ["London"] * 1000)
    total, correct = utils.evaluate_places(eval_corpus_path, predictions)

    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print("No target provided!")

    
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
