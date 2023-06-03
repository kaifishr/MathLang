# MathNet

Solving arithmetic expressions or simplifying complex algebraic expressions by expressing them in a more concise and manageable form requires a network to follow a certain set of rules and techniques that have to be learned during training and requires some form of reasoning capabilities.

This project investigates the capabilities of neural networks architectures to solve math problems and possible applications of pre-training on mathematical expressions for natural language processing and to test and predict network capabilities. Due to the interesting properties of mathematical expressions of arbitrary complexity, this project is potentially interesting to test neural network architectures designed for natural language processing.

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

- Both algebraic expressions and sentences are means of communication as they are designed to carry information and rely on synta and grammatical rules.
- Terms in an algebraic expression can be related to each other encouraging the network to detect these relationships.
- Algebraic expressions are well-suited to solve complex relationships between
term and encourage a neural network to set term into relation and to perform sequential reasoning as algebraic expression are often nested and need to be solved starting from the lowest level upwards.

## Simplifying Algebraic Expressions

What makes this task interesting is that mathematical expressions such as arithmetic
or algebraic expressions can be generated sufficiently fast with arbitrary lenght and complexity. The ease of generating arbitrary complex expressions makes it interesting for experiments where the capabilities of networks want to be tested.
Besides the high flexibility in generating mathematical expressions, they also test
the networks reasoning capabilities as simplifying or solving mathematical expressions requires to learn and follow a set of rules and techniques such as

- Combining terms that have the same variables raised to the same powers
- Distributing a term to each term inside parentheses (distributive property)
- Looking for common factors in terms and factoring them out. $4a+2b = 2(2a+b)$
- Simplifying exponents when they have the same base $a^2 * a^4 = a^6$
- Utilizing algebraic properties such as the distributive property, commutative property, associative property.
- Removing unnecessary parentheses if they are not needed.
- Learning to simplify expressions within parentheses first, then perform any exponents, followed by multiplication and division, and finally addition and subtraction.


## Research Questions

- Analyze loss, accuracy, entropy, perplexity as a function of network parameters, activations, processed tokens, FLOPS, complexity / lenght of expressions:
- How does the network perform if it has more time to "think"?
- How do meta layers perform?

## TODOs

- Use same position and token embedding for all models. 
- Add check so that input / output sequence does not exceed length.
- Try to avoid padding at output for simple arithmetic problems.
- Compute correct max input and output length for padding.
- Adjust loss function to ignore padding at output.
