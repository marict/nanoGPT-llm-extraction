import json
import shlex

from graphql import parse
from runpod_service import _bash_c_quote, _build_training_command, _create_docker_script


def test_docker_args_graphql_safe():
    """Ensure docker_args does not break GraphQL syntax after JSON escaping."""
    # Build a realistic training command the way start_cloud_training would.
    training_command = _build_training_command(
        "config/train_default.py --batch_size=32",
        keep_alive=True,
        note="dev",
        wandb_run_id="abc123",
        script_name="train.py",
    )

    script = _create_docker_script(training_command)
    docker_args = _bash_c_quote(script)

    # Embed via JSON dump to get proper escaping of quotes/backslashes
    docker_args_json = json.dumps(docker_args)
    query = f"mutation {{ dummy(dockerArgs: {docker_args_json}) }}"

    # Should parse without raising SyntaxError
    parse(query)
