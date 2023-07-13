<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-15 19:50:27
-->
# what is chatGPT and why it works?
## It's just adding one word at a time
chatGPt just generate one word within given the prefix sentence 
it just produce the probability of each word may be generate.

**templeture**: the method to select word is not based only on the high probability, sometimes choose the  `low pro` may get more intrersting creation.

## Where do the probabilites come from?
**statitics** reading a bunch of text, we can find that letters occur with their probabilities, `a`may always with `m` to get `am`, but less with `g` to get `ag`

by statitics, we can get the probability of the rule of letter combination, scaling this up, we can get the token rule of combination

in English, the number of comman words will be 40,000, it's difficult to get like n gram probability(n gram means give n word prefix, generate the n+1 word's probability)

**large language model** is made to solve this problem, it just estimate the probability instead of record all the probability of all the cases

## What is Model?
**Model** is a function to estimate the underlying law which unknown to us

we just use the data we known, assume a model, force the model to fit the data already known

## Models for Human-Like Tasks
some models can handle tasks for hunman-being such as image recognition, so chatGPT is possible to generate text sentences

## Neural net
use some kinds of nuron, connect them to construct a network like human brain

## Mahcine Learning, and the Training of Neural Nets
machine use a lot of samples to learn the rule underlying of the data. this is called machine learning

the training of Neural Nets is use the gradient backpropagation.

## The pratice and Lore of Neural Net Training 
how to decide the certain number of parameters accroding to the difficulties is a problem

## "Surely a Network that's Big Enough Can Do Anything"
if a netwrok is big enough and have ways to train it then actually it can do anything

## The Concept Of Embedding
reflect a word to a vector, such that the computer can handle these vectors

different words will be reflected to a word space, some words will relfect close since they may have same meaning

## Inside chatGPT
use transformer to handle the relationship between words

## Training of chatGPT
just use the simple back propagation to train this model

## Upper Training of ChatGPT
use prompt to make chatGPT try to do something it can but it doesn't know

GPT just generate words without knowing why it need to do that, then use instruction training, it knows what to do 

## What really make chatGPT do?
inside GPT, it's a great amount of parameters, with this scale of parameters, nothing will be impossible to generate a model

GPT may learn something very fundemental to language, such as semantic tree, logic stuff.

## Semantic motion in chatGPT
the sentence generation process can be viewd as a path in the word space of chatGPT, observe this path, may be we cna get some knoledge from it

all in all, chatGPT prove that language may not be so hard to understand, it can be cacluate by machine