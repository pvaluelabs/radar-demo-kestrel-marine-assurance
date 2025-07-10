import pytest
from unittest.mock import MagicMock
from kestrel_platform.orchestration.gateway import MultiModelOrchestrationGateway
from kestrel_platform.prompts.templates import SYSTEM_UNDERWRITING_PROMPT, RISK_EVALUATION_TEMPLATE

def test_prompt_template_formatting():
    """
    Test that the cargo transit template formats properly.
    """
    formatted = RISK_EVALUATION_TEMPLATE.format(
        vessel_name="Poseidon-IV",
        vessel_class="S-Class Carrier",
        dwt=72000,
        cargo_type="Perishable Foodstuffs",
        value="5,200,000",
        route_from="Rotterdam",
        route_to="New York"
    )
    assert "Poseidon-IV" in formatted
    assert "Rotterdam -> New York" in formatted
    assert "$5,200,000" in formatted

def test_gateway_model_grade_evaluation():
    """
    Ensures that the orchestration gateway is calling a current model string.
    This serves as a mock regression test to prove our prompt eval harness works end-to-end.
    """
    gateway = MultiModelOrchestrationGateway(
        anthropic_key="sk-ant-test",
        openai_key="sk-or-test"
    )
    
    # Mocking the client calls to assert we invoke current model strings
    gateway.anthropic_client.messages.create = MagicMock()
    gateway.anthropic_client.messages.create.return_value.content = [
        MagicMock(text="Approved with 1.5% Hull Deductible")
    ]
    
    prompt = RISK_EVALUATION_TEMPLATE.format(
        vessel_name="Ariadne-IX",
        vessel_class="A-Class",
        dwt=35000,
        cargo_type="Timber",
        value="1,200,000",
        route_from="Oslo",
        route_to="Hamburg"
    )
    
    response = gateway.generate_completion(SYSTEM_UNDERWRITING_PROMPT, prompt)
    
    # Verify we are calling the current claude-sonnet-5 model
    gateway.anthropic_client.messages.create.assert_called_once()
    call_args = gateway.anthropic_client.messages.create.call_args[1]
    assert call_args["model"] == "claude-sonnet-5"
    assert "Oslo -> Hamburg" in call_args["messages"][0]["content"]
    assert response == "Approved with 1.5% Hull Deductible"
