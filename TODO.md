1. fix LSTM decoder..?
1. multiple layers for LSTM
1. get plain autoencoder working
    1. add embeddings?
1. args for data augumentation rates
1. add hyperparameter tuning via 
    1. json file listing out hyperparameter 
1. regularization?
1. additional user params
    1. data augmentation rates
    1. learning rate
1. saving and loading models

### Parsing
- lowercase all words
- replace card names from text and flavor text with ~
- remove dashes, periods, semi-colons, colons
- keep colons (period?)
- split mana costs / tap functions 
- split on spaces, newlines
- remove apostrophes 
- skip multi faced cards
- transform numbers into words
- reject unset cards
- split types by dashes and space   

### Evaluation Procedure
- 5-folds cross validation for each model
- nested cross validation for hyperparmeter selection and model evaluation
- after model selection, retrain on whole dataset - w/ a hyperparameter optimization routine

### Flavor Text Parsing
- autoencoder: embedding inputs into lower dimensional vector space, important features are given more emphasis
- word embedding: similar words have similar encoding in a vector space
- configure embedding layer
- how to combine embeddings of different cards
- create models
    - plain autoencoder
    - lstm autoencoder
    - lstm autoencoder w/ attention
    - transformer autoencoder
 
https://sebastianraschka.com/faq/docs/evaluate-a-model.html
https://weina.me/nested-cross-validation/

https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/
https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
    http://yaronvazana.com/2019/09/28/training-an-autoencoder-to-generate-text-embeddings/
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
https://machinelearningmastery.com/lstm-autoencoders/
https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/

https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/
https://www.tensorflow.org/tutorials/text/transformer
https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b