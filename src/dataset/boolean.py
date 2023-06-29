"""Boolean dataset.

Generates boolean sequences and their solutions using Python's 
'eval()' method.

Typical usage example:
    
    dataset = BooleanDataset()
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


class BooleanDataset(IterableDataset):
    """Creates an iterable dataset of boolean expressions.

    For parameters such as

        num_terms = 4
        max_number = 10

    an example of an boolean expression to solve would be to
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
    relation_set = ["<", ">", "<=", ">=", "==", "!="]
    operator_set = ["and", "or"]  # , "not"]
    booleans_set = ["True", "False"]

    # These probabilities determine properties of expressions.
    p_second_term = 0.5
    p_set_brackets = 0.5
    p_append_right_or_left = 0.5

    def __init__(
        self,
        num_terms: int = 4,
    ) -> None:
        """Initializes an insance of AlgebraicDataset.
        """
        self.num_terms = num_terms
        self.scalars = list(map(str, range(self.max_number + 1)))
        self.relational = list(self.relation_set)
        self.operators = list(self.operator_set)
        self.booleans = list(self.booleans_set)

        # List of all characters used for expressions is comprised of scalars,
        # relational operators, booleans operators, brackets, and blank spaces 
        # for padding.
        chars = (
            self.scalars 
            + self.relational 
            + self.operators
            + self.booleans
            + ["(", ")"] + [" "]
        )

        # Lookup table for character-index-translation.
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.num_tokens = len(self.char_to_idx)
        print(f"{self.char_to_idx =}")
        print(f"{self.idx_to_char =}")

        self.max_input_length = self._comp_max_input_length()
        self.max_output_length = 2  # True or False

        print(f"Maximum input sequence lenght: {self.max_input_length}")
        print(f"Maximum output sequence lenght: {self.max_output_length}")

    def _comp_max_input_length(self) -> int:
        """Computes maximum input lenght for padding.

        Every logical expression is enclosed in brackets ((!(a&b))|c), where
        `a`, `b`, and `c` are boolean variables.

        Returns:
            Integer representing maximum length of input sequence.
        """

        # Compute maximum length of atomic term

        # Total lenght of opening and closing brackets.
        n_brackets = 2
        # Total length of operators.
        n_operators = 1
        # Total length of characters used to display scalars.
        n_scalars = 2 * len(str(self.max_number))

        n_max_term = n_brackets + n_operators + n_scalars

        # Operators used to concatenate terms.
        n_concat_operators = 1

        max_len_input = n_max_term + (self.num_terms - 1) * (
            n_max_term + n_concat_operators + n_brackets 
        )
        return max_len_input

    def generate_term(self) -> str:  # TODO: use better function names.
        """Generates random term."""
        term = random.choice(self.booleans)
        return f"({term})"

    def _get_term(self) -> str:
        """Generates random term of random length."""
        term = self.generate_term()
        if random.random() < self.p_second_term:
            term2 = self.generate_term()
            operator = random.choice(self.operators)
            return f"({term}{operator}{term2})"
        return f"({term})"

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
            result = str(eval(expression))

            print(f"{expression = }")
            print(f"{result = }")

            yield expression, result

            # # Add padding so that expressions and results have the same length.
            # expression = expression.ljust(self.max_input_length, " ")
            # result = result.ljust(self.max_output_length, " ")

            # # Encode expression and result using lookup table.
            # x_encoded = [self.char_to_idx[char] for char in expression]
            # y_encoded = [self.char_to_idx[char] for char in result]
            # x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            # y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            # yield x_data, y_data


def main():

    torch.manual_seed(42)
    random.seed(42)

    dataset = BooleanDataset(
        num_terms=8
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    for i, (x, y) in enumerate(dataloader):
        # print(f"{x = }")
        # print(f"{y = }")
        if i == 4:
            break

if __name__ == "__main__":
    main()
