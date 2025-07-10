# radar-demo-kestrel-marine-assurance

> [!IMPORTANT]
> **SYNTHETIC DEMO REPOSITORY — fictional, no real data**

This repository contains a synthetic modern AI-native insurance platform designed to demonstrate a high AI readiness profile for analytical scans.

## Project Features
- **Multi-Provider Orchestration**: Supports Anthropic and OpenAI integrations with automated dynamic fallback capabilities.
- **Dedicated Vector DB Client**: Integrates with modern vector databases (e.g., Qdrant) for high-dimensional semantic indexing.
- **ML Pipeline**: Trains specialized risk models utilizing target variables and rich historical features.
- **Prompt Evaluation & Testing**: Contains a robust `pytest` suite designed to evaluate prompts and output qualities programmatically against assertions.

## Project Structure
- `kestrel_platform/orchestration/`: Multi-provider router with API level failover mechanisms.
- `kestrel_platform/vectordb/`: Client connections and index management.
- `kestrel_platform/ml_pipeline/`: Data preprocessing, feature engineering, and predictive training scripts.
- `kestrel_platform/prompts/`: Versioned semantic prompt templates.
- `kestrel_platform/tests/`: Automated prompt testing and system validation suites.

## Installation
```bash
pip install -r requirements.txt
```

## Running Tests
```bash
pytest kestrel_platform/tests/
```
