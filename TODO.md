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

### Name Parsing

### Text Parsing
- remove reminder text

### Evaluation Procedure
- 5-folds cross validation for each model
- nested cross validation for hyperparmeter selection and model evaluation
- after model selection, retrain on whole dataset - w/ a hyperparameter optimization routine

### Flavor Text Parsing
- autoencoder: embedding inputs into lower dimensional vector space, important features are given more emphasis
- word embedding: similar words have similar encoding in a vector space

- seq2seq autoencoder  / embedding
- denoising autoencoder
- Take different vectors representing different parts of a card --> convert to fixed size embedding
convert each vector to fixed size, concatenate, convert to small size..?
- lstm autoencoder, lstm autoenoder with attention, transformer autoencoder

https://sebastianraschka.com/faq/docs/evaluate-a-model.html
https://weina.me/nested-cross-validation/
https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/
https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
http://yaronvazana.com/2019/09/28/training-an-autoencoder-to-generate-text-embeddings/
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
https://machinelearningmastery.com/lstm-autoencoders/
https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/