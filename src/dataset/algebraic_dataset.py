"""Arithmetic dataset.

Generates arithmetic sequences and their solution using Python's `eval()` method.

Typical usage example:
    TODO

"""
import collections
import random
import time

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset

from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify

torch.manual_seed(42)
random.seed(42)


class ArithmeticExpressionDataset(IterableDataset):
    """Creates an iterable dataset of arithmetic expressions.

    Creates arithmetical problems. For example, if the following parameters
    are used:

        len_expression = 4
        max_number = 10

    an example of an arithmetic expression to solve would be to
    '3+((8-9)+((9-0)+5))-(6+5)' which has the result 5.

    Attributes:
        len_expression:
        max_number:
        operators:
        scalars:
        chars:
        char_to_idx:
        idx_to_char:
        max_input_length:
        max_output_length:
    """

    max_number = 10
    operator_set = "+-"
    p_second_term = 0.5
    p_set_brackets = 0.5
    p_append_right_or_left = 0.5

    def __init__(
        self,
        len_expression: int = 4,
        max_input_length: int = None,
        max_output_length: int = None,
    ) -> None:
        """Initializes the arithmetic dataset based on provided parameters.

        Args:
            len_expression: An integer defining the number of iterations to create arithmetic expression.
            max_number: An integer defining the largest scalar value.
            operator_set: A string indicating which operator set to choose.
        """
        super().__init__()

        self.len_expression = len_expression

        # Operator set
        self.operators = list(self.operator_set)
        print(f"{self.operators = }")

        self.scalars = list(map(str, range(self.max_number)))

        chars = self.scalars + self.operators + ["(", ")"] + [" "]
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        print(f"{chars = }")
        print(f"{self.char_to_idx = }")
        print(f"{self.idx_to_char = }")

        self.max_input_length = (
            max_input_length if max_input_length else self._max_input_length()
        )
        self.max_output_length = (
            max_output_length if max_output_length else self._max_output_length()
        )
        print(f"{self.max_input_length = }")
        print(f"{self.max_output_length = }")

    def _max_input_length(self) -> int:
        """Computes maximum input lenght for padding.

        Returns:
            Maximum length of input.
        """
        max_len_term = (
            5  # Maximum length of a single term equals five characters: (a+b)
        )
        len_operator = 1
        len_brackets = 2
        return max_len_term + self.len_expression * (
            max_len_term + len_operator + len_brackets
        )

    def _max_output_length(self) -> int:
        """Computes maximum output lenght for padding.

        Returns:
            Maximum length of input.
        """
        # NOTE: Currently, the computation only works for the +- operators set as it does not consider multiplication.
        len_largest_result = len(
            str((1 + self.len_expression) * 2 * (self.max_number - 1))
        )
        len_unary_opeartor = 1
        return len_unary_opeartor + len_largest_result

    def _get_term(self) -> str:
        """Generates random term of random length."""
        term = random.choice(self.scalars)
        if random.random() < self.p_second_term:
            term2 = random.choice(self.scalars)
            operator = random.choice(self.operators)
            term = f"({term}{operator}{term2})"
        return term

    def generate_expression(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random arithmetic expression."""
        expression = collections.deque()
        expression.append(self._get_term())

        for _ in range(self.len_expression):
            term = self._get_term()
            operator = random.choice(self.operators)

            # Append term randomly either right or left.
            if random.random() < self.p_append_right_or_left:
                term = f"{operator}{term}"
                expression.append(term)
            else:
                term = f"{term}{operator}"
                expression.appendleft(term)

            # Set brackets randomly.
            if random.random() < self.p_set_brackets:
                expression.appendleft("(")
                expression.append(")")

        # Remove whitespaces
        expression = "".join(expression)

        return expression

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            expression = self.generate_expression()
            result = str(eval(expression))

            print(f"{expression = }")
            print(f"{len(expression) = }")
            print(f"{result = }")
            print(f"{len(result) = }")

            # Add padding so that expressions and results have always the same length.
            expression = expression.ljust(self.max_input_length, " ")
            result = result.ljust(self.max_output_length, " ")

            # TODO: Add padding to expression and result.
            # Encode expression and result using translation table.
            x_encoded = [self.char_to_idx[char] for char in expression]
            y_encoded = [self.char_to_idx[char] for char in result]
            x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            yield x_data, y_data


def main():
    print("\nArithmeticExpressionDataset\n")
    dataset = ArithmeticExpressionDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    for i, (x, y) in enumerate(dataloader):
        print(f"{x = }")
        print(f"{y = }")
        if i == 1:
            break


if __name__ == "__main__":
    main()
