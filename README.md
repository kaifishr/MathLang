# MathNet

## Motivation

Mathematical expressions, such as arithmetic or algebraic ones, can be generated quickly, at low cost, and with arbitrarily high complexity. Furthermore, these expressions come with short- and long-range dependencies, which makes them useful to train a network to learn long-range dependencies between tokens in a sequence.

Algebraic and arithmetic expressions are well-suited for learning how to resolve complex relationships between terms, encourage a neural network to set terms into relation with each other, and perform sequential reasoning as such expressions are often nested and need to be solved starting from the lowest level upwards.

Solving arithmetic expressions or simplifying complex algebraic expressions by expressing them in a more concise and manageable form requires following a certain set of rules and techniques that the network needs to learn during training.

For example, the network must learn that brackets determine the processing order of terms or that terms far apart in the sequence can cancel each other out. In more detail, simplifying or solving mathematical expressions requires the network to learn the distributive, commutative, and associative laws, to simplify expressions within parentheses first (if necessary), then perform any exponents, followed by multiplication and division, and finally addition and subtraction. Last but not least, the network needs to learn to remove redundant brackets and sort the simplified expression to arrive at its final form.

The ease of generating arbitrary complex mathematical expressions combined with their numerous interesting properties makes them interesting for experiments where the capabilities of networks or novel network architectures want to be tested.

## Algebraic and Arithmetic Expressions

An [algebraic expression](https://en.wikipedia.org/wikiAlgebraic_expression) is a [mathematical expression](https://en.wikipedia.org/wiki/Expression_(mathematics)) that follows a well-formed syntax, carries semantic meaning, consists of terms, variables, constants, and mathematical operations such as addition, subtraction, multiplication, division, and exponentiation, and can look as follows:

$$
((3 \cdot x-6 \cdot z)-((x-1)-((1 \cdot y-(((7+x)+(8-z))-5 \cdot y))+1)-z))
$$

For the expression above, the network's task would be to compute a simplified version of the expression that is $x+6*y-4*z-13$. The simplification is performed using *SymPy*'s `simplify()` method. Note that `simplify()` also sorts the variables and adds the scalar term at the end of the simplified expression.

Alebraic expressions are a superset of arithmetic expressions such as

$$
3+6-(1+3)+((8-3)+((7-9)-(9+0)-(6+2))+(1-6))
$$

where the network's task is to compute the scalar result $-14$. Here, Python's `eval()` method is used to solve the expression. For the sake of simplicity, only operators for addition, subtraction, and multiplication are used for the generation of expressions.

## Research Questions

- Analyze loss, accuracy, entropy, perplexity as a function of network parameters, activations, processed tokens, FLOPS, complexity / lenght of expressions:
- How does the network perform if it has more time to "think"?
    - Is it possible to "simulate" big networks.
- How do meta layers perform?
- Can mathematical expressions used for a pretraining strategy for langauge modles.

## TODOs

- Add extra folder with projects
    - calculator
    - experiments ...
- Compute correct max input and output length for padding.
- Add check so that input / output sequence does not exceed length.