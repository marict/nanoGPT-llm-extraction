import random
import re

from tiktoken import get_encoding

from data.dagset.streaming import generate_single_dag_example
from models.dag_model import OP_NAMES


def test_text_generation_rounding():
    """Generates 1000 dag examples without english conversion and checks that the string values
    are the example same as the target initial values"""
    for i in range(1000):
        rng = random.Random(i)
        example = generate_single_dag_example(
            depth=4, rng=rng, conversion_probability=0
        )
        text = example.text
        exp_initial_values = example.initial_values
        target_operations = example.operations
        # Convert operation vectors to readable names
        operation_names = [OP_NAMES[idx] for idx in target_operations.argmax(dim=1)]
        # Extract all numbers from the generated text
        string_numbers = re.findall(r"-?\d+\.\d+", text)
        # Check that the numbers are the same as the target initial values
        non_pad_initial_values = [v for v in exp_initial_values if v != 1.0]
        for number in string_numbers:
            assert (
                float(number) in exp_initial_values
            ), f"Seed {i} generated value mismatch. \nText: {text!r}, \nInitial values: {exp_initial_values!r}, \nOperations: {operation_names!r}"
        for number in non_pad_initial_values:
            assert (
                str(number) in string_numbers
            ), f"Seed {i} generated value mismatch. \nText: {text!r}, \nInitial values: {exp_initial_values!r}, \nOperations: {operation_names!r}, \nString numbers: {string_numbers!r}"


def test_text_generation_seed_1620():
    """Test seed 1620 that previously had "less" after parenthesis issue.

    This test locks the random number generator to seed 1620 that
    produced a problematic sample during validation. The assertion
    captures the fixed behaviour - if the bug returns, this test will fail.
    """
    rng = random.Random(1620)
    example = generate_single_dag_example(depth=4, rng=rng, conversion_probability=0.3)
    generated = example.text

    expected = (
        "2.8 divided by ( ( -169.0 / ( 7.46 - ( negative thirty-three + -628.0 ) ) ) )"
    )

    # Ensure the fixed text is generated (no "less" after parenthesis).
    assert (
        generated == expected
    ), f"Seed 1620 generated text differs. Got: {generated!r}"

    # Double-check the GPT-2 token count matches the observed 31 tokens (was 35 before fix).
    enc = get_encoding("gpt2")
    assert len(enc.encode(generated)) == 31


def test_text_generation_seed_2652():
    """Test seed 2652 that has incorrect initial values issue.

    This test captures the case where 112.84 appears in initial values
    when it should be 1.0 due to padding logic. The assertion intentionally
    captures the buggy behaviour so that future fixes will surface as a
    test failure, making the regression explicit.
    """
    rng = random.Random(2652)
    example = generate_single_dag_example(depth=4, rng=rng, conversion_probability=0.3)
    generated = example.text

    # Check that the initial values contain the problematic 112.84
    assert (
        112.84 not in example.initial_values
    ), f"Seed 2652 initial values contain 112.84. Got: {example.initial_values}"

    assert (
        len([i for i in example.initial_values if i == 1.0]) == 4
    ), f"Seed 2652 initial values don't contain correct number of 1.0s. Got: {example.initial_values}"

    # Check that operations are all identity as expected
    expected_ops = ["identity", "identity", "identity", "identity"]
    actual_ops = [OP_NAMES[i] for i in example.operations.argmax(dim=1)]
    assert actual_ops == expected_ops, f"Seed 2652 operations differ. Got: {actual_ops}"
