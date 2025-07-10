import random

from tiktoken import get_encoding

from data.dagset.streaming import generate_single_dag_example


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
    actual_ops = [
        ["add", "subtract", "multiply", "divide", "identity"][i]
        for i in example.operations.argmax(dim=1)
    ]
    assert actual_ops == expected_ops, f"Seed 2652 operations differ. Got: {actual_ops}"
