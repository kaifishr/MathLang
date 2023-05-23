# MathNet

Investigating the capabilities of neural networks to solve math problems and 
possible applications of pre-training on algebraic expressions for natural 
language processing.

## Algebraic Expressions and Sentences

An [algebraic expression](https://en.wikipedia.org/wikiAlgebraic_expression) is a [mathematical expression](https://en.wikipedia.org/wiki/Expression_(mathematics)) consisting of constants, variables, terms, and mathematical operations such as addition, subtraction, multiplication, division, and exponentiation and can look as follows:

$$4 + 2*a + b*(b - 3c)$$

Algebraic expressions follow a well-formed **syntax**, carry **semantic meaning**, and can be understood as **stentences**. Alebraic expressions are a superset of arithmetic expressions such as:

$$(4 + 2) + (1-4)*(9 - 3)$$

Algebraic expressions like the ones above provide a flexible way to manipulate and analyze mathematical **relationships**.

Algebraic expressions and sentences, like this one, share certain structural characteristics and the goal of conveying meaning.

A sentence is a grammatical unit of language that carries information and is composed of words in a structured manner following a **syntax**.

As algebraic expression, sentences can vary in length and complexity, and serve the purpose of transmitting information or expresssing thoughts or ideas.

From a structure point of view, both algebraic expressions and sentences have a structured format and they consist of a combination of elements arranged in a specific order to convey meaning.

## TODOs

- Add encoder-only transformer model.
- Try to avoid padding at output for simple arithmetic problems.
- Adjust loss function to ignore padding at output.
- Compute correct max input and output length for padding.
