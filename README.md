# Using Graph Neural Networks to predict the CPC labels of patents

This is an example f how to use Graph Neural Networks to predict the CPC labels of patents. The data is from the [Lens](https://www.lens.org/) database. The data is preprocessed and stored in a duckDB database. The data is stored in the `Data` folder. The data is stored in the following tables:

Handselected:
- antiSeed
- metalTextiles
- naturalTextiles
- mineralTextiles
- syntheticTextiles

10000 samples filtered by CPC classes and keywords, but not manually checked:
mineralTextilesTestSet (From the mineralTextiles)

## Known Issues
- We have Empty CPC classes, we need to remove them (E.g. "CPC Class: """ --> This makes it very akward to work with the data, but in theory it should work)     
- The keyphrase Extractor has limited capabilities and has been made for another usecase. See more in the [keyphraseExtractors](../keyphraseExtractors) folder.

## How to run
1. Install the requirements (Currently not fully documented)
- torch
- duckDB
2. Run the `main.py` file


There is a lot of commented code in the different files for activation different features. The code is not cleaned up, but it should be easy to follow.
