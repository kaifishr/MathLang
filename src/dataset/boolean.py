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
    operator_set = ["and", "or"]
    negation_set = ["not"]
    booleans_set = ["True", "False"]

    # These probabilities determine properties of expressions.
    p_second_term = 0.5
    p_negate_term = 0.5
    p_set_brackets = 0.5
    p_boolean_term = 0.5
    p_append_right = 0.5

    def __init__(self, num_terms: int = 4) -> None:
        """Initializes an insance of AlgebraicDataset."""

        self.num_terms = num_terms
        self.scalars = list(map(str, range(self.max_number + 1)))
        self.relational = list(self.relation_set)
        self.operators = list(self.operator_set)
        self.booleans = list(self.booleans_set)
        self.negation = list(self.negation_set)

        # List of all characters used for expressions is comprised of scalars,
        # relational operators, booleans operators, brackets, and blank spaces
        # for padding.
        chars = (
            self.scalars
            + self.relational
            + self.operators
            + self.booleans
            + self.negation_set
            + ["(", ")"]
            + [" "]
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

    def _get_boolean_term(self) -> str:
        """Generates random term."""
        term = random.choice(self.booleans)
        if random.random() < self.p_negate_term:
            return f"({self.negation[0]} {term})"
        return f"({term})"

    def _generate_boolean_term(self) -> str:
        """Generates random term of random length."""
        term = self._get_boolean_term()
        if random.random() < self.p_second_term:
            term2 = self._get_boolean_term()
            operator = random.choice(self.operators)
            if random.random() < self.p_negate_term:
                return f"({self.negation[0]}({term}{operator}{term2}))"
            return f"({term}{operator}{term2})"
        return f"({term})"

    def _generate_relational_term(self) -> str:
        """Generates random relational term of random length."""
        term = random.choice(self.scalars)
        if random.random() < self.p_second_term:
            term_2 = random.choice(self.scalars)
            operator = random.choice(self.relational)
            return f"({term}{operator}{term_2})"
        return f"({term})"

    def _generate_expression(self) -> str:
        """Generates random algebraic expression."""
        expression = collections.deque()

        # Append initial term.
        if random.random() < self.p_boolean_term:
            term = self._generate_boolean_term()
        else:
            term = self._generate_relational_term()

        expression.append(term)

        for _ in range(self.num_terms - 1):

            if random.random() < self.p_boolean_term:
                term = self._generate_boolean_term()
            else:
                term = self._generate_relational_term()

            operator = random.choice(self.operators)

            # Append term randomly either right or left.
            if random.random() < self.p_append_right:
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
            expression = self._generate_expression()
            result = str(bool(eval(expression)))
            exit()

            # TODO: Work here with tokens in a list instad of characters in a string.

            # Add padding to ensure all inputs have same length. 
            expression += [" "] * (self.max_input_length - len(expression))
            print(f"{expression = }")
            print(f"{len(expression) = }")
            expression = expression.ljust(self.max_input_length, " ")

            print(f"{expression = }")
            print(f"{result = }")

            # Encode expression and result using lookup table.
            x_encoded = [self.char_to_idx[char] for char in expression]
            y_encoded = [self.char_to_idx[char] for char in result]
            x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            yield x_data, y_data


def main():
    torch.manual_seed(42)
    random.seed(42)

    dataset = BooleanDataset(num_terms=4)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    for i, (x, y) in enumerate(dataloader):
        print(f"{x = }")
        print(f"{y = }")
        if i == 10:
            break


if __name__ == "__main__":
    main()
