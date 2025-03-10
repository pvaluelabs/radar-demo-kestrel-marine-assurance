import anthropic
import openai
import logging

logger = logging.getLogger(__name__)

class MultiModelOrchestrationGateway:
    """
    Enterprise-grade multi-provider gateway.
    Routes queries to Anthropic's claude-sonnet-5 as primary,
    and falls back to OpenAI's gpt-4o if primary fails or is slow.
    """
    def __init__(self, anthropic_key: str, openai_key: str):
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        self.openai_client = openai.OpenAI(api_key=openai_key)

    def generate_completion(self, system_prompt: str, prompt: str) -> str:
        # Primary try: Anthropic Claude Sonnet 5 (Current model string)
        try:
            logger.info("Attempting primary generation via Anthropic claude-sonnet-5...")
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-5",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic gateway failed: {e}. Executing fallback to OpenAI...")
            
            # Fallback try: OpenAI GPT-4o (Current model string)
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as fe:
                logger.error(f"All AI providers failed: {fe}")
                raise RuntimeError(f"Orchestration gateway failed: {fe}")
