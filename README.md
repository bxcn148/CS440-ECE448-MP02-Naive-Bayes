Download link :https://programming.engineering/product/cs440-ece448-mp02-naive-bayes/

# CS440-ECE448-MP02-Naive-Bayes
CS440/ECE448 MP02: Naive Bayes
The first thing you need to do is to download mp02.zip. Content is similar to MP01.

mp02_notebook.ipynb

This file ( ) will walk you through the whole MP, giving you instructions and debugging tips as you go.

Table of Contents

Reading the Data

Learning a Naive Bayes Model: Maximum Likelihood

Learning a Naive Bayes Model: Stop Words

Learning a Naive Bayes Model: Laplace Smoothing

Decisions Using a Naive Bayes Model

Optimizing Hyperparameters

Grade Your Homework

Reading the data

The dataset in your template package consists of 10000 positive and 3000 negative movie reviews. It is a subset of the Stanford Movie Review Dataset, which was originally introduced by this paper. We have split this data set for you into 5000 development examples and 8000 training examples. The autograder also has a hidden set of test examples, generally similar to the development dataset.

The data folder is structured like this:

├─ train

├─ neg

│ └─ 2000 negative movie reviews (text)

└─ pos

└─ 6000 positive movie reviews (text)

└─ dev

├─ neg

└─ 1000 negative movie reviews (text) └─ pos

└─ 4000 positive movie reviews (text)

In order to help you load the data, we provide you with a utility function called

reader.py

This has two new functions that didn’t exist in mp01:



loadTrain: load a training set



loadDev: load a dev set

2/1/24, 7:04 PM mp02_notebook


2/1/24, 7:04 PM mp02_notebook

In order to understand Naive Bayes, it might be useful to know the difference between bigram types and bigram tokens.



Bigram token: A bigram token consists of two consecutive word tokens from the

text. For grading purpose, all bigrams are represented as word1*-*-*-*word2 , i.e. *-*-*-* as a separator, and not as a tuple. In the context of bigrams, the tokens are these pairs of words. To emphasize, note that our definition of bigram is two “consecutive” word tokens. Consecutive!



Bigram type: Bigram types are the set of unique pairs of consecutive words that

occurred in a text. The number of bigram types in the nth positive text can be found by first generating all bigram tokens from the text and then counting the unique

bigrams.

A Naive Bayes model consists of two types of probability distributions:

The prior is the distribution over classes, P (Class).



The likelihood is the probability of a bigram token given a particular class,

(Bigram|Class).

The prior can be estimated from the training data. In your training data,

P (Class = pos) = 0.75.

Often, though, the testing data will have a different class distribution than the training data. If you don’t know the testing priors, then it’s sometimes best to just assume a

uniform distribution, i.e., P (Class = pos) = 0.5.

The likelihood is the informative part of a Naive Bayes model: it tells you which words are used more often in negative versus positive movie reviews.

There are many ways in which you can estimate the likelihood. The following formula is called the maximum likelihood estimate, because it maximizes the likelihood of the

words in your training dataset:

\# tokens of bigram x in texts of class y P (Bigram = x|Class = y) = \# tokens of any bigram in texts of class y


2/1/24, 7:04 PM mp02_notebook

Help on function create_frequency_table in module submitted:

create_frequency_table(train)

Parameters:

train (dict of list of lists)

– train[y][i][k] = k’th token of i’th text of class y

Output:

frequency (dict of Counters):

– frequency[y][x] = number of occurrences of bigram x in texts of c

lass y,

where x is in the format ‘word1*-*-*-*word2’

create_frequency_table

Edit so that it does what its docstring says it should do.


Hint: your code will be shorter if you use the python data structure called a Counter.

When your code works, you should get the following results:


2/1/24, 7:04 PM mp02_notebook

frequency[‘pos’][(‘this’, ‘film’)]= 2656

frequency[‘neg’][(‘this’, ‘film’)]= 879

frequency[‘pos’][(‘the’, ‘movie’)]= 2507

frequency[‘neg’][(‘the’, ‘movie’)]= 1098

frequency[‘pos’][(‘of’, ‘the’)]= 10008

frequency[‘neg’][(‘of’, ‘the’)]= 2779

frequency[‘pos’][(‘to’, ‘be’)]= 2500

frequency[‘neg’][(‘to’, ‘be’)]= 1057

frequency[‘pos’][(‘and’, ‘the’)]= 3342

frequency[‘neg’][(‘and’, ‘the’)]= 983

————————————–

Total # tokens in pos texts is 1421513

Total # tokens in neg texts is 468194

Total # types in pos texts is 475651

Total # types in neg texts is 195021

Learning a Naive Bayes model: Stop words

There are many common word pairs (bigrams) that may seem to be unrelated to whether a movie review is positive or negative. Due to the nature of the training data, it’s possible that some bigrams, consisting entirely of common words like “is”, “of”, “and”, etc., are more frequent in one part of the training data than another. This can be problematic, as it means a test review might be wrongly classified as “positive” just because it contains many instances of innocuous bigrams like (“and”, “the”).

A “stopword list” is a list of words that should not be considered when you classify a test

text. In the context of bigrams, we consider a bigram as a stopword if both of its

constituent words are in the stopword list. There are many candidate stopword lists

available on the internet. The stopword list that we’ve provided for you is based on this

one: https://www.ranks.nl/stopwords.

To emphasize, we consider a bigram as a stopword if “both” of its constituent words are

in the stopword list. Both!

Here is our stopword list:


2/1/24, 7:04 PM mp02_notebook

[“‘d”, “‘ll”, “‘m”, “‘re”, “‘s”, “‘t”, “‘ve”, ‘a’, ‘about’, ‘above’, ‘afte r’, ‘again’, ‘against’, ‘all’, ‘am’, ‘an’, ‘and’, ‘any’, ‘are’, ‘aren’, ‘a s’, ‘at’, ‘be’, ‘because’, ‘been’, ‘before’, ‘being’, ‘below’, ‘between’, ‘both’, ‘but’, ‘by’, ‘can’, ‘cannot’, ‘could’, ‘couldn’, ‘did’, ‘didn’, ‘d o’, ‘does’, ‘doesn’, ‘doing’, ‘don’, ‘down’, ‘during’, ‘each’, ‘few’, ‘fo r’, ‘from’, ‘further’, ‘had’, ‘hadn’, ‘has’, ‘hasn’, ‘have’, ‘haven’, ‘havi ng’, ‘he’, ‘her’, ‘here’, ‘hers’, ‘herself’, ‘him’, ‘himself’, ‘his’, ‘ho w’, ‘i’, ‘if’, ‘in’, ‘into’, ‘is’, ‘isn’, ‘it’, ‘its’, ‘itself’, ‘let’, ‘l l’, ‘me’, ‘more’, ‘most’, ‘mustn’, ‘my’, ‘myself’, ‘no’, ‘nor’, ‘not’, ‘o f’, ‘off’, ‘on’, ‘once’, ‘only’, ‘or’, ‘other’, ‘ought’, ‘our’, ‘ours’, ‘ou rselves’, ‘out’, ‘over’, ‘own’, ‘same’, ‘shan’, ‘she’, ‘should’, ‘shouldn’, ‘so’, ‘some’, ‘such’, ‘than’, ‘that’, ‘the’, ‘their’, ‘theirs’, ‘them’, ‘th emselves’, ‘then’, ‘there’, ‘these’, ‘they’, ‘this’, ‘those’, ‘through’, ‘t o’, ‘too’, ‘under’, ‘until’, ‘up’, ‘very’, ‘was’, ‘wasn’, ‘we’, ‘were’, ‘we ren’, ‘what’, ‘when’, ‘where’, ‘which’, ‘while’, ‘who’, ‘whom’, ‘why’, ‘wit h’, ‘won’, ‘would’, ‘wouldn’, ‘you’, ‘your’, ‘yours’, ‘yourself’, ‘yourselv es’]

To effectively avoid counting bigrams that are considered stopwords, two steps are necessary:

Pretend that the frequency of bigrams, where both words are stopwords, in the training corpus is zero.

. Ignore such bigrams if they occur in testing data.

In this part of the MP, you should set the frequencies of those bigram stopwords to zero.

del


Use the command (see Counters), so that these bigrams don’t get counted among either the bigram types or the bigram tokens.


2/1/24, 7:04 PM mp02_notebook

print(“frequency[‘pos’][(‘of’, ‘the’)]=”,frequency[‘pos’][‘of*-*-*-*the’])


print(“frequency[‘pos’][(‘of’, ‘the’)]=”,nonstop[‘pos’][‘of*-*-*-*the’])

print(“\n”)

print(“frequency[‘pos’][(‘to’, ‘be’)]=”,frequency[‘pos’][‘to*-*-*-*be’])

print(“frequency[‘pos’][(‘to’, ‘be’)]=”,nonstop[‘pos’][‘to*-*-*-*be’])

print(“\n”)

print(“frequency[‘pos’][(‘and’, ‘the’)]=”,frequency[‘pos’][‘and*-*-*-*the’])

print(“frequency[‘pos’][(‘and’, ‘the’)]=”,nonstop[‘pos’][‘and*-*-*-*the’])

print(“\n”)

print(“————————————–\n”)

print(“Total pos frequency:”,sum(frequency[‘pos’].values()))

print(“Total pos non-stopwords”,sum(nonstop[‘pos’].values()))

print(“\n”)

print(“Total # types in pos texts is”,len(frequency[‘pos’].keys()))

print(“Total # non-stopwords in pos is”,len(nonstop[‘pos’].keys()))

print(“Length of the stopwords set is:”,len(submitted.stopwords))

frequency[‘pos’][(‘this’, ‘film’)]= 2656

frequency[‘pos’][(‘this’, ‘film’)]= 2656

frequency[‘pos’][(‘the’, ‘movie’)]= 2507

frequency[‘pos’][(‘the’, ‘movie’)]= 2507

frequency[‘pos’][(‘of’, ‘the’)]= 10008

frequency[‘pos’][(‘of’, ‘the’)]= 0

frequency[‘pos’][(‘to’, ‘be’)]= 2500

frequency[‘pos’][(‘to’, ‘be’)]= 0

frequency[‘pos’][(‘and’, ‘the’)]= 3342

frequency[‘pos’][(‘and’, ‘the’)]= 0

————————————–

Total pos frequency: 1421513

Total pos non-stopwords 1168682

Total # types in pos texts is 475651

Total # non-stopwords in pos is 468246

Length of the stopwords set is: 150

Learning a Naive Bayes model: Laplace Smoothing

The maximum likelihood formula results in some words having zero probability, just

because they were not contained in your training data. A better formula is given by

Laplace smoothing, according to which

(\# tokens of bigram x in texts of clas


P (Bigram = x|Class = y) =

(\# tokens of any bigram in texts of class y) + k × (\#

2/1/24, 7:04 PM mp02_notebook

…where k is a hyperparameter that is usually chosen by trying several different values, and choosing the value that gives you the best accuracy on your development dataset.

+1


The in the denominator is used to account for bigrams that were never seen in the

training dataset for class y. All such words are mapped to the type

OOV

(out of

vocabulary), which has the likelihood

2/1/24, 7:04 PM mp02_notebook

Help on function naive_bayes in module submitted:

naive_bayes(texts, likelihood, prior)

Parameters:

texts (list of lists) –

texts[i][k] = k’th token of i’th text likelihood (dict of dicts)

likelihood[y][x] = Laplace-smoothed likelihood of bigram x given

y,

where x is in the format ‘word1*-*-*-*word2’ prior (float)

– prior = the prior probability of the class called “pos”

Output:

hypotheses (list)

– hypotheses[i] = class label for the i’th text

reader.loadDev

Use to load the dev set, then try classifying it with, say, a prior of

0.5:


2/1/24, 7:04 PM mp02_notebook

If you’ve reached this point, and all of the above sections work, then you’re ready to try grading your homework! Before you submit it to Gradescope, try grading it on your own machine. This will run some visible test cases (which you can read in

tests/test_visible.py

), and compare the results to the solutions (which you can

solution.json

read in ).

The exclamation point (!) tells python to run the following as a shell command. Obviously you don’t need to run the code this way — this usage is here just to remind you that you can also, if you wish, run this command in a terminal window.


