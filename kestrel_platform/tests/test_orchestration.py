import pytest
from unittest.mock import MagicMock
from kestrel_platform.orchestration.gateway import MultiModelOrchestrationGateway
from kestrel_platform.prompts.templates import SYSTEM_UNDERWRITING_PROMPT

def test_orchestration_fallback_behavior():
    """
    Test that the orchestration gateway falls back to OpenAI if Anthropic fails.
    And verify that the fallback call uses a current model string (gpt-4o).
    """
    gateway = MultiModelOrchestrationGateway(
        anthropic_key="sk-ant-test",
        openai_key="sk-or-test"
    )
    
    # Force Anthropic to fail to test fallback
    gateway.anthropic_client.messages.create = MagicMock(side_effect=Exception("Anthropic API Overloaded"))
    
    # Mock OpenAI client
    gateway.openai_client.chat.completions.create = MagicMock()
    gateway.openai_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Fallback answer: Approved"))
    ]
    
    response = gateway.generate_completion(SYSTEM_UNDERWRITING_PROMPT, "Analyze cargo profile: wood")
    
    # Verify fallback happened
    gateway.anthropic_client.messages.create.assert_called_once()
    gateway.openai_client.chat.completions.create.assert_called_once()
    
    openai_call_args = gateway.openai_client.chat.completions.create.call_args[1]
    assert openai_call_args["model"] == "gpt-4o"
    assert response == "Fallback answer: Approved"
