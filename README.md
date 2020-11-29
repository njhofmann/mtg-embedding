# mtg-embeddings

Project exploring the creation of low dimensional embeddings for
*Magic: The Gathering* cards, for the purposes of computing numerical similarity and and visualizations.

## Motivation

*Magic: The Gathering* has been a passion and hobby of mine for many years, and over the past several months I have been 
very interested in exploring autoencoders as a concept and the various implementations that have been discovered.

This project aims to solve the latter by investigating the task of creating low-dimensional embeddings for MTG cards, a
very high dimensional set of objects.

This project does not use raw images of MTG cards, opting instead to use the textual data making up a card's 
converted mana cost, type line, text, and any flavor text (of course keep the card name for reference). Since the 
dimensionality of each different aspect being considered differents, autoencoder models are evaluated separately for 
each aspects (plain, LSTM, and transformer based autoencoders with various hyperparameter setups).

This means a separate encoder (and thus set of embeddings) is created for converted mana costs, type lines, card texts, 
and flavor texts. To create a single embedding for a MTG card, these separate embeddings are concatenated together.

These final embeddings can (hopefully!) then be compared using cosine similarity, Euclidean distance, etc.

## Set Up

Install the necessary libraries under `requirements.txt` with `pip` wit Python 3.8.

## Running 

First need to get the latest batch of MTG cards, this project uses the "Oracle Cards" bulk data dump from the 
[Scryfall API](https://scryfall.com/docs/api/bulk-data). Download and extract to the top level directory of the project.

Next create the necessary data files by running `python -m src.gather_cards [name-of-data-dump.json]`.

Then encoders can be created for each type of data, with user entered arguments & hyperparameters 

Finally low-dimensional embeddings can be created for each MTG card with `python -m src.embeddings_main [...args...]`. 
These can be used in various comparison setups and visualizations (such as t-SNE).

Using the help flag `-h` for any script will give more information about what arguments are needed and available.
