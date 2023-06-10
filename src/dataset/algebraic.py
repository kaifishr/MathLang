"""Algebraic dataset.

Generates algebraic expressions along with their simplified versions using 
SymPy.

Typical usage example:

    dataset = AlgebraicDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for x, y in dataloader:
        pass  # Do stuff with `x` and `y`.
"""
import collections
import random

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset

from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify


class AlgebraicDataset(IterableDataset):
    """Creates an iterable dataset of algebraic expressions.

    Creates algebraic expressions. For example, if the following parameters are 
    used:

        num_terms = 4
        max_number = 10

    an example of an algebraic expression to solve would be to
    "9*b-((d+b)-a+0*b)+(a-d)" with the result "2*a+8*b-2*d".

    Attributes:
        num_terms: Integer defining number of terms of algebraic expression.
        use_simplify: Boolean. If true, algebraic expression is being
            additionally simplified.
        ooperators: List of operators used to build algebraic expression.
        variables: List of variables used to build algebraic expression.
        scalars: List of scalars used to build algebraic expression.
        chars:
        char_to_idx:
        idx_to_char:
    """

    max_number = 9
    operator_set = ["+", "-"]
    variable_set = ["x", "y", "z"]

    # These probabilities determine properties of expressions.
    p_scalar_term = 0.5
    p_scalar_multiplier = 0.5
    p_second_term = 0.5
    p_set_brackets = 0.5
    p_append_right_or_left = 0.5

    def __init__(
        self,
        num_terms: int = 4,
        use_simplify: bool = False,
    ) -> None:
        """Initializes an insance of AlgebraicDataset.
        """
        self.num_terms = num_terms
        self.use_simplify = use_simplify 
        self.operators = list(self.operator_set)
        self.variables = list(self.variable_set)
        self.scalars = list(map(str, range(self.max_number + 1)))

        # List of all characters used for expressions is comprised of scalar 
        # values, variables, operators, brackets, and blank spaces for padding.
        chars = (
            self.scalars 
            + self.operators 
            + self.variables 
            + ["*"] + ["(", ")"] + [" "]
        )

        # Lookup table for character-index-translation.
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.num_tokens = len(self.char_to_idx)

        self.max_input_length = self._comp_max_input_length()
        self.max_output_length = self._comp_max_output_length()

    def _comp_max_input_length(self) -> int:
        """Computes maximum input lenght for padding.

        To determine the maximum input length we assume expressions consisting
        only of addition or multiplication operations (depending on the set of
        operations choosen). 
        
        For scalars 0 to 9, and variables [a, b], the maximum length of an
        atomic term is nine characters: (2*a+3*b)
                                        123456789

        Returns:
            Integer representing maximum length of input sequence.
        """

        # Compute maximum length of atomic term

        # Total lenght of opening and closing brackets.
        n_brackets = 2
        # Total length of operators.
        n_operators = 3
        # Total lenght of single characer variables.
        n_variables = 2
        # Total length of characters used to display scalars.
        n_scalars = 2 * len(str(self.max_number))

        n_max_term = n_brackets + n_operators + n_variables + n_scalars

        # Operators used to concatenate terms.
        n_concat_operators = 1

        max_len_input = n_max_term + (self.num_terms - 1) * (
            n_max_term + n_concat_operators + n_brackets 
        )
        return max_len_input

    def _comp_max_output_length(self) -> int:
        """Computes maximum output lenght required for padding.
        
        Ok, here we are lazy because the weather is good and we want to go
        outside. We just assume that the simplified output is never longer 
        than the input.

        Returns:
            Integer representing maximum length of output sequence.
        """
        return self._comp_max_input_length()

    def generate_term(self) -> str:  # TODO: use better function names.
        """Generates random term."""
        if random.random() < self.p_scalar_term:
            term = random.choice(self.scalars)
        else:
            term = random.choice(self.variables)
            if random.random() < self.p_scalar_multiplier:
                scalar = random.choice(self.scalars)
                term = f"{scalar}*{term}"
        return term

    def _get_term(self) -> str:
        """Generates random term of random length."""
        term = self.generate_term()
        if random.random() < self.p_second_term:
            term2 = self.generate_term()
            operator = random.choice(self.operators)
            term = f"({term}{operator}{term2})"
        return term

    def generate_expression(self) -> str:
        """Generates random algebraic expression."""
        expression = collections.deque()

        # Append initial term.
        expression.append(self._get_term())

        for _ in range(self.num_terms - 1):
            term = self._get_term()
            operator = random.choice(self.operators)

            # Append term randomly either right or left.
            if random.random() < self.p_append_right_or_left:
                term = f"{operator}{term}"
                expression.append(term)
            else:
                term = f"{term}{operator}"
                expression.appendleft(term)

            # Whether or not to set brackets.
            if random.random() < self.p_set_brackets:
                expression.appendleft("(")
                expression.append(")")

        # Remove whitespaces
        expression = "".join(expression)

        return expression

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            expression = self.generate_expression()
            result = str(parse_expr(expression, evaluate=True))
            if self.use_simplify:
                result = simplify(result)
            result = str(result).replace(" ", "")

            # Add padding so that expressions and results are always of the 
            # same length.
            expression = expression.ljust(self.max_input_length, " ")
            result = result.ljust(self.max_output_length, " ")

            # Encode expression and result using translation table.
            x_encoded = [self.char_to_idx[char] for char in expression]
            y_encoded = [self.char_to_idx[char] for char in result]
            x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            yield x_data, y_data


def main():

    torch.manual_seed(42)
    random.seed(42)

    dataset = AlgebraicDataset(
        num_terms=8
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for i, (x, y) in enumerate(dataloader):
        print(f"{x = }")
        print(f"{y = }")
        if i == 3:
            break

if __name__ == "__main__":
    main()
