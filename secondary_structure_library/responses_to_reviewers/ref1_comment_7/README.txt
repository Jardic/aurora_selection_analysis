In this folder I answer the question about the number of seuqneces in the training and testing sets. 

I think the referee was not very specific, so there are two tables here to look at:
    - 'split_sizes_ML_easy.csv' which shows the numbers for different samplings in the random-split - ML-easy pipeline. This is the one where no sequences are generated.

    - 'split_sizes_ML_hard.csv' which shows the same numbers but for the pipeline which attempts to predict the top seuqneces. Note, that here, the test and validation sets are the same for all samplings, its just the training data which gets sampled and so changes.