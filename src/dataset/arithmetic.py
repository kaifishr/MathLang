"""Arithmetic dataset.

Generates arithmetic sequences and their solution using Python's 
'eval()' method.

Typical usage example:

    dataset = ArithmeticDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for x, y in dataloader:
        print(f"{x = }")
        print(f"{y = }")
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
    p_second_term = 0.5
    p_set_brackets = 0.5
    p_append_right_or_left = 0.5

    def __init__(
        self,
        num_terms: int = 8,
        max_input_length: int = None,
        max_output_length: int = None,
    ) -> None:
        """Initializes the arithmetic dataset based on provided parameters.

        Args:
            num_terms: An integer defining the number of iterations to create arithmetic expression.
            max_number: An integer defining the largest scalar value.
            operators: A string indicating which operator set to choose.
            scalars:
        """
        super().__init__()

        self.num_terms = num_terms
        self.operators = list(self.operator_set)
        self.scalars = list(map(str, range(self.max_number + 1)))
        print(f"{self.scalars = }")

        # List of all characters used for arithmetic expressions is comprised
        # of scalar values, operators, brackets, and blank spaces for padding.
        chars = self.scalars + self.operators + ["(", ")"] + [" "]

        # Lookup table for character-index-translation.
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

        self.max_input_length = (
            max_input_length if max_input_length else self._max_input_length()
        )
        self.max_output_length = (
            max_output_length if max_output_length else self._max_output_length()
        )

        self.num_tokens = len(self.char_to_idx)

        print(f"Maximum input sequence lenght: {self.max_input_length}")
        print(f"Maximum output sequence lenght: {self.max_output_length}")

    def _max_input_length(self) -> int:
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
        n_operator = 1  # Lenght of operator (+, -, *).
        n_brackets = 2  # Lenght of opening and closing brackets.
        n_max_term = n_brackets + n_operator + 2 * len(str(self.max_number))
        return n_max_term + self.num_terms * (
            n_max_term + n_operator + n_brackets 
        )

    def _max_output_length(self) -> int:
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

    def generate_expression(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random arithmetic expression."""
        expression = collections.deque()
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
        num_terms=1,
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

    for i, (x, y) in enumerate(dataloader):
        print(f"{x.shape = }")
        print(f"{y = }")
        if i == 2:
            break


if __name__ == "__main__":
    main()