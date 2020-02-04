# Movie-Rating-NLP-
1. Dataset:- 
 
a) The Movie Review Data is a collection of movie reviews retrieved from the imdb.com. 
 
b) The dataset is comprised of 1,000 positive and 1,000 negative movie reviews. 
 
c) Reviews are stored one per file with a naming convention cv000to cv999 for each neg and pos. 
 
2. Data Preparation:- 
 
a) Separation of data into training and test sets. 
 
b) Loading and cleaning the data to remove punctuation and numbers.  
 
c) Defining a vocabulary of preferred words 
 
3. Split into Train and Test Sets:- 
 
a) In this system, it can predict the sentiment of a textual movie review/comment as either positive or negative or give him rating out of 5 according to this. 
 
b) In this, the last 100 positive reviews and the last 100 negative reviews are used as a test set (100 reviews) and the remaining 1,800 reviews as the training dataset. 
 
c) This is a 90% train, 10% split of the data. 
 
d) The split can be imposed easily by using the filenames of the reviews where reviews named 000 to 899 are for training data and reviews named 900 onwards are for testing the model. 
 
4. Preprocessing data:- 
 
a) Loading and cleaning review data 
1. Split tokens on white space. 
2. Remove all punctuation from words. 
3. Remove all words that are not purely comprised of alphabetical characters. 
4. Remove all words that are known as stop words. 
5. Remove all words that have a length <= 1   character. 
