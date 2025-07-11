import random
import re

from data.dagset.streaming import generate_single_dag_example
from models.dag_model import OP_NAMES


def test_text_generation_rounding():
    """Generates 1000 dag examples without english conversion and checks that the string values
    are the example same as the target initial values"""
    for i in range(1000):
        example = generate_single_dag_example(depth=4, conversion_probability=0)
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
            # Handle the special case of -0.0 vs 0.0 in string representation
            number_str = str(number)
            if number_str == "-0.0":
                number_str = "0.0"  # -0.0 appears as 0.0 in text
            assert (
                number_str in string_numbers
            ), f"Seed {i} generated value mismatch. \nText: {text!r}, \nInitial values: {exp_initial_values!r}, \nOperations: {operation_names!r}, \nString numbers: {string_numbers!r}"
