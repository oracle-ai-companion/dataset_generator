# Dataset Generator

This project is a powerful tool for generating multi-turn conversational datasets using various AI models. It's designed to create high-quality, diverse datasets suitable for fine-tuning language models or testing conversational AI systems.

## Features

- Generate single-turn or multi-turn conversations
- Support for multiple AI models (gemini-pro, gemini-1.5-pro, gemini-1.5-flash)
- Customizable number of turns and conversations
- Output in JSON or JSONL format
- Fine-tuning format support
- Batch processing capabilities
- Asynchronous generation for improved performance

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/dataset-generator.git
   cd dataset-generator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your API key:
   ```
   API_KEY=your_api_key_here
   ```

## Usage

### Running the FastAPI Server

1. Start the FastAPI server:
   ```
   python -m src.api.main
   ```

2. The server will start running on `http://localhost:8000`.

### Generating Datasets

You can generate datasets using the `/generate/multi-turn` endpoint. Here's an example using curl:

```bash
curl -X POST "http://localhost:8000/generate/multi-turn" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "Let'\''s have a conversation about artificial intelligence.",
         "num_turns": 3,
         "model": "gemini-1.5-flash",
         "fine_tuning_format": true,
         "output_file": "output/ai_conversation.jsonl",
         "num_conversations": 5
     }'
```

This will generate 5 conversations, each with 3 turns, about artificial intelligence using the gemini-1.5-flash model. The output will be saved in the fine-tuning format to `output/ai_conversation.jsonl`.

### API Parameters

- `prompt`: The initial prompt to start the conversation.
- `num_turns`: Number of turns in each conversation.
- `model`: The AI model to use (options: "gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash").
- `fine_tuning_format`: Whether to output in a format suitable for fine-tuning (default: false).
- `output_file`: The file to save the generated conversations (optional).
- `num_conversations`: Number of conversations to generate (default: 1).

## Output Format

The generated dataset will be in JSONL format, with each line containing a JSON object representing a complete conversation. Here's an example of the structure:

```py
{
  "messages": [
    {"role": "user", "content": "Let's talk about AI."},
    {"role": "assistant", "content": "Sure! What aspect of AI interests you the most?"},
    {"role": "user", "content": "I'm curious about its potential impact on jobs."}
  ],
  "metadata": {
    "num_turns": 3,
    "model": "gemini-1.5-flash",
    "conversation_index": 0
  }
}
```

## Batch Processing

For batch processing, you can use the `generate_batch_dataset` method in the `DatasetGenerator` class. This allows you to generate multiple datasets with different prompts and parameters in a single run.

## Error Handling and Logging

The application includes comprehensive error handling and logging. Check the console output for debug information and any error messages during the generation process.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
