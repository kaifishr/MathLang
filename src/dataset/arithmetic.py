"""Arithmetic dataset.

Generates arithmetic sequences and their solution using Python's `eval()`
method.

Typical usage example:

    dataset = ArithmeticDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for x, y in dataloader:
        pass  # Do stuff with `x` and `y`.
"""
import collections
import random

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset


class ArithmeticDataset(IterableDataset):
    """Creates an iterable dataset of arithmetic expressions.

    Creates arithmetical problems. For example, if the following parameters
    are used:

        num_terms = 4
        max_number = 10

    an example of an arithmetic expression to solve would be to
    '3+((8-9)+((9-0)+5))-(6+5)' which has the result 5.

    Attributes:
        num_terms:
        max_number:
        operators:
        scalars:
        chars:
        char_to_idx:
        idx_to_char:
        max_input_length:
        max_output_length:
    """
    max_number = 9
    operator_set = ["+", "-"]

    # These probabilities determine properties of expressions.
    p_second_term = 0.5
    p_set_brackets = 0.5
    p_append_right = 0.5

    def __init__(
        self,
        num_terms: int = 4,
    ) -> None:
        """Initializes the arithmetic dataset based on provided parameters.

        Args:
            num_terms: An integer defining the number of iterations to create 
                arithmetic expression.
        """
        super().__init__()

        self.num_terms = num_terms
        self.operators = list(self.operator_set)
        self.scalars = list(map(str, range(self.max_number + 1)))

        # List of all characters used for arithmetic expressions is comprised
        # of scalar values, operators, brackets, and blank spaces for padding.
        chars = self.scalars + self.operators + ["(", ")"] + [" "]

        # Lookup table for character-index-translation.
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.num_tokens = len(self.char_to_idx)

        self.max_input_length = self._comp_max_input_length()
        self.max_output_length = self._comp_max_output_length()

        print(f"Maximum input sequence lenght: {self.max_input_length}")
        print(f"Maximum output sequence lenght: {self.max_output_length}")

    def _comp_max_input_length(self) -> int:
        """Computes maximum input lenght for padding.

        To determine the maximum input length we assume expressions consisting
        only of addition or multiplication operations (depending on the set of
        operations choosen).

        For scalars 0 to 9, the maximum length of a single term is five
        characters: (a+b)
                    12345

        Returns:
            Maximum length of input.
        """
        # Total lenght of opening and closing brackets.
        n_brackets = 2
        # Total length of operators.
        n_operator = 1
        # Total length of characters used to display scalars.
        n_scalars = 2 * len(str(self.max_number))

        n_max_term = n_brackets + n_operator + n_scalars
        max_len_input = n_max_term + (self.num_terms - 1) * (
            n_max_term + n_operator + n_brackets
        )
        return max_len_input

    def _comp_max_output_length(self) -> int:
        """Computes maximum output lenght for padding.

        Returns:
            Maximum length of input.
        """
        len_unary_opeartor = 1
        if "*" in self.operator_set:
            max_result = self.max_number ** (2 * self.num_terms)
        else:
            max_result = self.num_terms * (2 * self.max_number)
        max_result_len = len(str(max_result))
        return len_unary_opeartor + max_result_len

    def _get_term(self) -> str:
        """Generates random term of random length."""
        term = random.choice(self.scalars)
        if random.random() < self.p_second_term:
            term2 = random.choice(self.scalars)
            operator = random.choice(self.operators)
            term = f"({term}{operator}{term2})"
        return term

    def _generate_expression(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random arithmetic expression."""
        expression = collections.deque()
        expression.append(self._get_term())

        for _ in range(self.num_terms - 1):
            term = self._get_term()
            operator = random.choice(self.operators)

            # Append term randomly either right or left.
            if random.random() < self.p_append_right:
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
            expression = self._generate_expression()
            result = str(eval(expression))

            # Add padding so that expressions and results have the same length.
            expression = expression.ljust(self.max_input_length, " ")
            result = result.ljust(self.max_output_length, " ")

            # Encode expression and result using lookup table.
            x_encoded = [self.char_to_idx[char] for char in expression]
            y_encoded = [self.char_to_idx[char] for char in result]
            x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            yield x_data, y_data


def main():
    torch.manual_seed(42)
    random.seed(42)

    dataset = ArithmeticDataset(
        num_terms=8,
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

    for i, (x, y) in enumerate(dataloader):
        print(f"{x.shape = }")
        print(f"{y = }")
        if i == 2:
            break


if __name__ == "__main__":
    main()
