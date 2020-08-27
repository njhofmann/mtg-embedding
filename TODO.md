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

### Flavor Text Parsing

# Emebedding
- seq2seq autoencoder 
- Take different vectors representing different parts of a card --> convert to fixed size embedding
- convert each vector to fixed size, concatenate, convert to small size..?