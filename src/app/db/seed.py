"""
Scalable database seeding with configurable quantities.

Usage:
    # Default (small dataset for quick dev)
    await seed_if_empty()

    # Large dataset (10,000+ tools)
    await seed_if_empty(SeedConfig(tool_count=10000))
"""
import random
from dataclasses import dataclass
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models import Provider, Category, Tool, Parameter, Example


@dataclass
class SeedConfig:
    """Configuration for scalable seeding."""
    tool_count: int = 7  # Default small for quick dev
    params_per_tool: tuple[int, int] = (1, 5)  # min, max parameters per tool
    examples_per_tool: tuple[int, int] = (0, 3)  # min, max examples per tool
    batch_size: int = 1000  # Insert batch size for large datasets


# =============================================================================
# Seed Sample Data (templates for random generation)
# =============================================================================

PROVIDERS_SAMPLE = [
    {"name": "OpenAI", "logo_url": "https://openai.com/logo.png", "website": "https://openai.com"},
    {"name": "Anthropic", "logo_url": "https://anthropic.com/logo.png", "website": "https://anthropic.com"},
    {"name": "Google", "logo_url": "https://google.com/logo.png", "website": "https://google.com"},
    {"name": "Meta", "logo_url": "https://meta.com/logo.png", "website": "https://meta.com"},
    {"name": "Microsoft", "logo_url": "https://microsoft.com/logo.png", "website": "https://microsoft.com"},
    {"name": "Amazon", "logo_url": "https://aws.amazon.com/logo.png", "website": "https://aws.amazon.com"},
    {"name": "Cohere", "logo_url": "https://cohere.com/logo.png", "website": "https://cohere.com"},
    {"name": "Stability AI", "logo_url": "https://stability.ai/logo.png", "website": "https://stability.ai"},
    {"name": "Mistral", "logo_url": "https://mistral.ai/logo.png", "website": "https://mistral.ai"},
    {"name": "Hugging Face", "logo_url": "https://huggingface.co/logo.png", "website": "https://huggingface.co"},
]

CATEGORIES_SAMPLE = [
    {"name": "Text Generation", "description": "Tools for generating text content including articles, summaries, and creative writing"},
    {"name": "Code Generation", "description": "Tools for generating, analyzing, and refactoring code across multiple languages"},
    {"name": "Image Analysis", "description": "Tools for analyzing, describing, and extracting information from images"},
    {"name": "Data Extraction", "description": "Tools for extracting structured data from unstructured text and documents"},
    {"name": "Translation", "description": "Tools for translating text between languages with context awareness"},
    {"name": "Sentiment Analysis", "description": "Tools for analyzing sentiment, emotion, and tone in text"},
    {"name": "Question Answering", "description": "Tools for answering questions based on provided context or knowledge"},
    {"name": "Summarization", "description": "Tools for condensing long documents into concise summaries"},
    {"name": "Classification", "description": "Tools for categorizing text, images, or data into predefined classes"},
    {"name": "Embedding", "description": "Tools for generating vector embeddings for semantic search and similarity"},
    {"name": "Speech Processing", "description": "Tools for speech-to-text, text-to-speech, and audio analysis"},
    {"name": "Document Processing", "description": "Tools for parsing, extracting, and understanding documents"},
]

TOOL_NAMES_SAMPLE = [
    # Text Generation
    "Text Summarizer", "Blog Writer", "Email Composer", "Article Generator", "Story Writer",
    "Headline Generator", "Product Description Writer", "Ad Copy Creator", "Social Media Post Generator",
    "Newsletter Writer", "Press Release Generator", "Speech Writer", "Slogan Creator",
    # Code
    "Code Reviewer", "SQL Generator", "Unit Test Generator", "Code Refactorer", "Bug Detector",
    "API Generator", "Documentation Generator", "Code Translator", "Regex Generator",
    "Schema Generator", "Query Optimizer", "Code Explainer", "Snippet Generator",
    # Analysis
    "Image Captioner", "Object Detector", "Face Analyzer", "Scene Classifier", "OCR Extractor",
    "Chart Reader", "Logo Detector", "Color Analyzer", "Style Classifier",
    # Data
    "JSON Extractor", "Table Parser", "Entity Extractor", "Resume Parser", "Invoice Extractor",
    "Receipt Scanner", "Form Filler", "Data Normalizer", "Schema Mapper",
    # NLP
    "Sentiment Analyzer", "Topic Classifier", "Language Detector", "Keyword Extractor",
    "Named Entity Recognizer", "Intent Classifier", "Spam Detector", "Toxicity Filter",
    # Translation
    "Universal Translator", "Legal Translator", "Medical Translator", "Technical Translator",
    # QA
    "FAQ Bot", "Document QA", "Knowledge Base Search", "Contextual Answerer",
    # Embedding
    "Text Embedder", "Image Embedder", "Multi-modal Embedder", "Semantic Search",
    # Speech
    "Speech Transcriber", "Voice Synthesizer", "Accent Detector", "Speaker Identifier",
    # Document
    "PDF Parser", "Contract Analyzer", "Report Generator", "Meeting Summarizer",
]

TOOL_DESCRIPTIONS_SAMPLE = [
    "Processes input data and generates high-quality output using advanced AI models",
    "Analyzes content to extract meaningful insights and structured information",
    "Transforms unstructured data into organized, actionable formats",
    "Leverages deep learning to understand context and generate relevant responses",
    "Automates repetitive tasks with intelligent pattern recognition",
    "Provides real-time analysis with high accuracy and low latency",
    "Combines multiple AI techniques for comprehensive processing",
    "Offers customizable parameters for fine-tuned results",
    "Scales efficiently from small to enterprise workloads",
    "Integrates seamlessly with existing workflows and APIs",
    "Supports multiple languages and regional variations",
    "Maintains context across long documents and conversations",
    "Produces human-quality output with minimal post-editing",
    "Handles edge cases gracefully with fallback mechanisms",
    "Optimized for both speed and accuracy in production environments",
]

VERSIONS_SAMPLE = [
    "1.0.0", "1.0.1", "1.1.0", "1.2.0", "1.5.0",
    "2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.5.0",
    "3.0.0", "3.1.0", "3.2.0", "4.0.0", "5.0.0",
]

PARAMETER_SAMPLES = [
    {"name": "text", "param_type": "string", "required": True, "description": "The input text to process"},
    {"name": "content", "param_type": "string", "required": True, "description": "The content to analyze"},
    {"name": "query", "param_type": "string", "required": True, "description": "The query or question to answer"},
    {"name": "code", "param_type": "string", "required": True, "description": "The source code to process"},
    {"name": "document", "param_type": "string", "required": True, "description": "The document content"},
    {"name": "url", "param_type": "string", "required": False, "description": "URL to fetch content from"},
    {"name": "max_length", "param_type": "integer", "required": False, "description": "Maximum output length"},
    {"name": "min_length", "param_type": "integer", "required": False, "description": "Minimum output length"},
    {"name": "temperature", "param_type": "float", "required": False, "description": "Creativity level (0.0-1.0)"},
    {"name": "top_p", "param_type": "float", "required": False, "description": "Nucleus sampling parameter"},
    {"name": "language", "param_type": "string", "required": False, "description": "Target language code"},
    {"name": "source_language", "param_type": "string", "required": False, "description": "Source language code"},
    {"name": "format", "param_type": "string", "required": False, "description": "Output format: json, text, markdown"},
    {"name": "style", "param_type": "string", "required": False, "description": "Output style: formal, casual, technical"},
    {"name": "tone", "param_type": "string", "required": False, "description": "Tone: professional, friendly, neutral"},
    {"name": "model", "param_type": "string", "required": False, "description": "Model variant to use"},
    {"name": "max_tokens", "param_type": "integer", "required": False, "description": "Maximum tokens in response"},
    {"name": "include_metadata", "param_type": "boolean", "required": False, "description": "Include processing metadata"},
    {"name": "confidence_threshold", "param_type": "float", "required": False, "description": "Minimum confidence score"},
    {"name": "batch_size", "param_type": "integer", "required": False, "description": "Items per batch"},
    {"name": "timeout", "param_type": "integer", "required": False, "description": "Request timeout in seconds"},
    {"name": "retry_count", "param_type": "integer", "required": False, "description": "Number of retries on failure"},
    {"name": "schema", "param_type": "string", "required": False, "description": "Expected output schema"},
    {"name": "context", "param_type": "string", "required": False, "description": "Additional context for processing"},
    {"name": "instructions", "param_type": "string", "required": False, "description": "Custom instructions"},
]

EXAMPLE_TITLES_SAMPLE = [
    "Basic Usage", "Advanced Example", "Quick Start", "Production Example",
    "Simple Case", "Complex Scenario", "Edge Case Handling", "Batch Processing",
    "Real-world Application", "Integration Example", "API Usage", "CLI Usage",
    "Error Handling", "Custom Configuration", "Performance Optimized",
]

EXAMPLE_INPUTS_SAMPLE = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "Climate change poses significant challenges to global ecosystems and human societies worldwide.",
    "def calculate_total(items): return sum(item.price for item in items)",
    "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY name",
    "The meeting discussed Q3 targets, budget allocation, and upcoming product launches.",
    "Please analyze the following data and provide insights on customer behavior patterns.",
    "Translate the following technical documentation to Spanish while maintaining accuracy.",
    "Extract all email addresses and phone numbers from the provided text.",
    "Summarize the key points from this 10-page research paper on machine learning.",
    "Generate unit tests for the authentication module with edge case coverage.",
    "Review this code for potential security vulnerabilities and performance issues.",
    "Create a product description for a wireless Bluetooth headphone with noise cancellation.",
    "Write a professional email declining a meeting invitation politely.",
    "Analyze the sentiment of customer reviews for the past quarter.",
    "Convert this JSON schema to a TypeScript interface definition.",
]

EXAMPLE_OUTPUTS_SAMPLE = [
    "Successfully processed input with 98.5% confidence score.",
    "Analysis complete: Identified 3 key themes and 12 supporting points.",
    "Generated 5 test cases covering happy path and 3 edge cases.",
    "Translation completed with cultural adaptations for target audience.",
    "Extracted 15 entities: 8 persons, 4 organizations, 3 locations.",
    "Summary: 3 main points in 150 words, preserving key statistics.",
    "Code review: Found 2 potential issues, 3 optimization suggestions.",
    "Sentiment analysis: 72% positive, 18% neutral, 10% negative.",
    "Classification: Category A (confidence: 0.94), Category B (confidence: 0.82).",
    "Generated embedding vector with 1536 dimensions.",
    "Refactored code: Reduced complexity from O(n²) to O(n log n).",
    "Documentation generated: 12 sections, 45 code examples.",
    "Query optimized: Execution time reduced from 2.3s to 0.15s.",
    "Parsed document: 24 pages, 156 paragraphs, 12 tables extracted.",
    "Response generated in 0.8 seconds with 512 tokens.",
]


# =============================================================================
# Seed Functions
# =============================================================================

def generate_tools(count: int, provider_ids: list[int], category_ids: list[int]) -> list[dict]:
    """Generate random tools from sample data."""
    tools = []
    for _ in range(count):
        tools.append({
            "name": random.choice(TOOL_NAMES_SAMPLE),
            "description": random.choice(TOOL_DESCRIPTIONS_SAMPLE),
            "version": random.choice(VERSIONS_SAMPLE),
            "provider_id": random.choice(provider_ids),
            "category_id": random.choice(category_ids),
            "usage_count": random.randint(100, 100000),
            "is_active": random.random() > 0.1,  # 90% active
        })
    return tools


def generate_parameters(tool_ids: list[int], config: SeedConfig) -> list[dict]:
    """Generate random parameters for tools."""
    parameters = []
    for tool_id in tool_ids:
        param_count = random.randint(*config.params_per_tool)
        # Ensure at least one required parameter
        selected_params = random.sample(PARAMETER_SAMPLES, min(param_count, len(PARAMETER_SAMPLES)))
        for i, param in enumerate(selected_params):
            parameters.append({
                "tool_id": tool_id,
                "name": param["name"],
                "param_type": param["param_type"],
                "required": param["required"] if i > 0 else True,  # First param always required
                "description": param["description"],
                "default_value": None if param["required"] else "auto",
            })
    return parameters


def generate_examples(tool_ids: list[int], config: SeedConfig) -> list[dict]:
    """Generate random examples for tools."""
    examples = []
    for tool_id in tool_ids:
        example_count = random.randint(*config.examples_per_tool)
        for _ in range(example_count):
            examples.append({
                "tool_id": tool_id,
                "title": random.choice(EXAMPLE_TITLES_SAMPLE),
                "input_text": random.choice(EXAMPLE_INPUTS_SAMPLE),
                "output_text": random.choice(EXAMPLE_OUTPUTS_SAMPLE),
            })
    return examples


async def seed_if_empty(config: SeedConfig | None = None) -> bool:
    """
    Seed database with data if empty.

    Args:
        config: Seeding configuration. Defaults to small dataset.

    Returns:
        True if seeding occurred, False if data already exists.
    """
    if config is None:
        config = SeedConfig()

    async with AsyncSessionLocal() as session:
        # Check if data exists
        result = await session.execute(select(Provider).limit(1))
        if result.scalar():
            return False  # Already seeded

        # Seed providers and categories (always use full sample)
        providers = [Provider(**p) for p in PROVIDERS_SAMPLE]
        categories = [Category(**c) for c in CATEGORIES_SAMPLE]

        session.add_all(providers)
        session.add_all(categories)
        await session.flush()

        provider_ids = [p.id for p in providers]
        category_ids = [c.id for c in categories]

        # Generate and insert tools in batches
        tool_data = generate_tools(config.tool_count, provider_ids, category_ids)
        tool_ids = []

        for i in range(0, len(tool_data), config.batch_size):
            batch = tool_data[i:i + config.batch_size]
            tools = [Tool(**t) for t in batch]
            session.add_all(tools)
            await session.flush()
            tool_ids.extend([t.id for t in tools])

            if config.tool_count >= 1000 and (i + config.batch_size) % 5000 == 0:
                print(f"  Seeded {i + config.batch_size} tools...")

        # Generate and insert parameters in batches
        param_data = generate_parameters(tool_ids, config)
        for i in range(0, len(param_data), config.batch_size):
            batch = param_data[i:i + config.batch_size]
            session.add_all([Parameter(**p) for p in batch])
            await session.flush()

        # Generate and insert examples in batches
        example_data = generate_examples(tool_ids, config)
        for i in range(0, len(example_data), config.batch_size):
            batch = example_data[i:i + config.batch_size]
            session.add_all([Example(**e) for e in batch])
            await session.flush()

        await session.commit()

        total_params = len(param_data)
        total_examples = len(example_data)
        print(f"Seeded: {len(providers)} providers, {len(categories)} categories, "
              f"{config.tool_count} tools, {total_params} parameters, {total_examples} examples")

        return True


# =============================================================================
# CLI for manual seeding with custom config
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import argparse
    import time

    from app.core.database import engine
    from app.models import Base

    parser = argparse.ArgumentParser(description="Seed database with sample data")
    parser.add_argument("--tools", type=int, default=10000, help="Number of tools to generate")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    args = parser.parse_args()

    config = SeedConfig(tool_count=args.tools, batch_size=args.batch_size)

    async def main():
        # Create tables first
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print(f"Seeding database with {config.tool_count} tools...")
        start = time.perf_counter()
        result = await seed_if_empty(config)
        elapsed = time.perf_counter() - start

        if result:
            print(f"Completed in {elapsed:.2f} seconds")
        else:
            print("Database already seeded. Delete dev.db to reseed.")

        await engine.dispose()

    asyncio.run(main())
