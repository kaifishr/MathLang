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
        num_terms: Integer defining number of terms of expression.
        scalars: List of scalars used to build algebraic expression.
        relational: List of relational relational operators.
        operators: List of boolean operators.
        booleans: List of boolean variables.
        negation: The negation operator.
        tokens: List of tokens.
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
        tokens = (
            self.scalars
            + self.relational
            + self.operators
            + self.booleans
            + self.negation_set
            + ["(", ")"]
            + [" "]
        )

        # Lookup table for character-index-translation.
        self.char_to_idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx_to_char = {idx: token for idx, token in enumerate(tokens)}
        self.num_tokens = len(self.char_to_idx)
        print(f"{self.char_to_idx =}")
        print(f"{self.idx_to_char =}")

        self.max_input_length = self._comp_max_input_length()
        self.max_output_length = 1  # True or False

        print(f"Maximum input sequence lenght: {self.max_input_length}")
        print(f"Maximum output sequence lenght: {self.max_output_length}")

    def _comp_max_input_length(self) -> int:
        """Computes maximum input lenght for padding.

        Computes an upper bound based on the maximum length of an atomic term:
            (not((not bool)operator(not bool))) e.g., (!((!0)&(!1)))

        Returns:
            Integer representing maximum length of input sequence.
        """
        n_max_term_atomic = 16  # (not((not bool)operator(not bool)))
        n_brackets = 2
        n_concat_operators = 1

        max_len_input = n_max_term_atomic + (self.num_terms - 1) * (
            n_max_term_atomic + n_concat_operators + n_brackets
        )

        return max_len_input

    def _get_boolean_term(self) -> str:
        """Generates random term."""
        term = random.choice(self.booleans)
        if random.random() < self.p_negate_term:
            negation = self.negation[0]
            return ["(", negation, " ", term, ")"]
        return ["(", term, ")"]

    def _generate_boolean_term(self) -> str:
        """Generates random term of random length."""
        term = self._get_boolean_term()
        if random.random() < self.p_second_term:
            term_2 = self._get_boolean_term()
            operator = random.choice(self.operators)
            if random.random() < self.p_negate_term:
                negation = self.negation[0]
                return ["(", negation, "(", *term, operator, *term_2, ")", ")"]
            return ["(", *term, operator, *term_2, ")"]
        return ["(", *term, ")"]

    def _generate_relational_term(self) -> str:
        """Generates random relational term of random length."""
        term = random.choice(self.scalars)
        if random.random() < self.p_second_term:
            term_2 = random.choice(self.scalars)
            operator = random.choice(self.relational)
            return ["(", term, operator, term_2, ")"]
        return ["(", term, ")"]

    def _generate_expression(self) -> str:
        """Generates random algebraic expression."""
        expression = []

        # Append initial term.
        if random.random() < self.p_boolean_term:
            term = self._generate_boolean_term()
        else:
            term = self._generate_relational_term()

        expression.extend(term)

        for _ in range(self.num_terms - 1):

            if random.random() < self.p_boolean_term:
                term = self._generate_boolean_term()
            else:
                term = self._generate_relational_term()

            operator = random.choice(self.operators)

            # Append term randomly either right or left.
            if random.random() < self.p_append_right:
                term = [operator, *term]
                expression = expression + term
            else:
                term = [*term, operator]
                expression = term + expression

            # Whether or not to set brackets.
            if random.random() < self.p_set_brackets:
                expression = ["("] + expression
                expression = expression + [")"]

        return expression

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:

        while True:

            expression = self._generate_expression()
            result = bool(eval("".join(expression)))

            # Add padding to ensure all inputs have same length. 
            expression += [" "] * (self.max_input_length - len(expression))

            # Encode expression and result using lookup table.
            x_encoded = [self.char_to_idx[char] for char in expression]
            y_encoded = [1 if result else 0]

            x_data = torch.tensor(data=x_encoded, dtype=torch.long)
            y_data = torch.tensor(data=y_encoded, dtype=torch.long)

            yield x_data, y_data


def main():
    torch.manual_seed(42)
    random.seed(42)

    dataset = BooleanDataset(num_terms=8)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=2)
    for i, (x, y) in enumerate(dataloader):
        print(f"{x.shape = }")
        print(f"{y.shape = }")
        if i == 10:
            break


if __name__ == "__main__":
    main()
