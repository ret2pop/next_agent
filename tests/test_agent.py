import os

from unittest.mock import patch, MagicMock

# Assuming the class LocalAgent and required imports are in next_agent.agent
# However since next_agent.agent has many side-effects in its __init__, we can mock what we need
# or we can test the method directly if possible.

from next_agent.agent import LocalAgent

@patch('next_agent.agent.tool_registry', MagicMock())
@patch('next_agent.agent.command_registry', MagicMock())
@patch('next_agent.agent.CodebaseRAG', MagicMock())
@patch('next_agent.agent.ChatOllama', MagicMock())
@patch('next_agent.agent.KokoroProvider', MagicMock())
@patch('next_agent.agent.FasterWhisperProvider', MagicMock())
@patch('next_agent.agent.get_app', MagicMock())
@patch('next_agent.agent.InMemoryHistory', MagicMock())
@patch('next_agent.agent.KeyBindings', MagicMock())
def test_open_in_emacs_no_shell():
    # To instantiate LocalAgent without executing its full init behavior that might require system paths,
    # we mock out the dependencies.
    agent = LocalAgent()

    test_filepath = "/tmp/test_file.txt"
    abs_path = os.path.abspath(test_filepath)

    with patch('next_agent.agent.subprocess.run') as mock_run:
        # Configure the mock to return a completed process with a 0 return code
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        agent._open_in_emacs(test_filepath)

        # Verify subprocess.run was called exactly once
        mock_run.assert_called_once()

        # Get the arguments it was called with
        args, kwargs = mock_run.call_args

        # Check that it's called with a list, not a string
        cmd_arg = args[0]
        assert isinstance(cmd_arg, list), f"Expected command to be a list, but got {type(cmd_arg)}"
        assert cmd_arg == ["emacsclient", "-n", abs_path]

        # Check that shell=True is NOT in the kwargs
        assert "shell" not in kwargs or not kwargs["shell"], "subprocess.run should not be called with shell=True"

        # Verify other kwargs
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
