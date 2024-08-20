import os
import asyncio
import aiofiles
import json
from typing import List, Dict, Any, AsyncGenerator, Union
from src.gemini_client import GeminiClient

class DatasetGenerator:
    def __init__(self):
        self.client = GeminiClient()

    async def _generate_content(self, prompt: str, model: str) -> str:
        try:
            return await self.client.generate_response(prompt, model)
        except Exception as e:
            print(f"Error generating content: {e}")
            raise

    async def generate_single_turn_dataset_stream(
        self, prompt: str, num_samples: int, model: str, fine_tuning_format: bool, output_file: str
    ) -> AsyncGenerator[str, None]:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create output directory if it doesn't exist
        async with aiofiles.open(output_file, mode='w') as f:
            for i in range(num_samples):
                response = await self._generate_content(prompt, model)
                data = {
                    "prompt": prompt,
                    "response": response,
                    "metadata": {"sample_id": i, "model": model}
                }
                if fine_tuning_format:
                    data = self._format_for_fine_tuning(data)
                json_data = json.dumps(data) + "\n"
                await f.write(json_data)
                yield json_data
                await asyncio.sleep(0.1)  # Rate limiting

    async def generate_multi_turn_dataset_stream(
        self, prompt: str, num_turns: int, num_samples: int, model: str, fine_tuning_format: bool, output_file: str
    ) -> AsyncGenerator[str, None]:
        for _ in range(num_samples):
            conversation = []
            current_prompt = prompt

            for turn in range(num_turns):
                # Generate input (human message)
                if turn == 0:
                    input_content = current_prompt
                else:
                    input_content = await self._generate_content(current_prompt, model)
                
                input_data = {"input": {"content": input_content}}
                conversation.append(input_data)
                
                # Generate output (AI response) based on the input
                output_prompt = f"{current_prompt}\n\nHuman: {input_content}\n\nAI:"
                output_content = await self._generate_content(output_prompt, model)
                output_data = {"output": {"content": output_content}}
                conversation.append(output_data)
                
                # Update the prompt for the next turn
                current_prompt = f"{output_prompt} {output_content}"

            yield json.dumps(conversation) + "\n"

    async def generate_batch_dataset(
        self, requests: List[Union[Dict[str, Any], Any]], output_file: str, model: str, fine_tuning_format: bool
    ):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        async with aiofiles.open(output_file, mode='w') as f:
            for idx, request in enumerate(requests):
                if isinstance(request, dict):
                    prompt = request.get("prompt")
                    num_samples = request.get("num_samples")
                    num_turns = request.get("num_turns")
                else:
                    prompt = request.prompt
                    num_samples = request.num_samples
                    num_turns = getattr(request, "num_turns", None)

                model_data = {
                    "id": idx,
                    "model": model,
                    "instances": [],
                    "metadata": {"num_turns": num_turns if num_turns else 1}
                }

                if num_turns:
                    async for conversation_data in self.generate_multi_turn_dataset_stream(
                        prompt, num_turns, num_samples, model, fine_tuning_format, output_file
                    ):
                        conversation = json.loads(conversation_data)
                        model_data["instances"].append({
                            "input": {"content": prompt},
                            "output": {"content": conversation}
                        })
                else:
                    async for data in self.generate_single_turn_dataset_stream(
                        prompt, num_samples, model, fine_tuning_format, output_file
                    ):
                        instance = json.loads(data)
                        model_data["instances"].append(instance)

                await f.write(json.dumps(model_data, indent=4) + '\n')

    def _format_for_fine_tuning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the generated data for fine-tuning.
        
        This method prepares the data in a format suitable for fine-tuning language models.
        It handles both single-turn and multi-turn conversations.
        """
        formatted_data = {
            "messages": []
        }
        for turn in data["conversation"]:
            if "input" in turn:
                formatted_data["messages"].append({
                    "role": "user",
                    "content": turn["input"]["content"]
                })
            elif "output" in turn:
                formatted_data["messages"].append({
                    "role": "assistant",
                    "content": turn["output"]["content"]
                })
        formatted_data["metadata"] = data["metadata"]
        return formatted_data

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the generated data to ensure it meets the required format and quality standards.
        
        Returns True if the data is valid, False otherwise.
        """
        try:
            # Check for required keys
            required_keys = ["metadata"]
            if not all(key in data for key in required_keys):
                return False

            # Validate metadata
            if not isinstance(data["metadata"], dict):
                return False
            if "sample_id" not in data["metadata"] or "model" not in data["metadata"]:
                return False

            # Validate conversation or prompt/response pair
            if "conversation" in data:
                if not isinstance(data["conversation"], list) or len(data["conversation"]) == 0:
                    return False
                for turn in data["conversation"]:
                    if not isinstance(turn, dict) or "prompt" not in turn or "response" not in turn:
                        return False
                    if not isinstance(turn["prompt"], str) or not isinstance(turn["response"], str):
                        return False
            else:
                if "prompt" not in data or "response" not in data:
                    return False
                if not isinstance(data["prompt"], str) or not isinstance(data["response"], str):
                    return False

            # Validate content length
            max_length = 4096  # Adjust this value based on your model's requirements
            if "conversation" in data:
                for turn in data["conversation"]:
                    if len(turn["prompt"]) > max_length or len(turn["response"]) > max_length:
                        return False
            else:
                if len(data["prompt"]) > max_length or len(data["response"]) > max_length:
                    return False

            # Add any additional validation checks here
            # For example, you might want to check for minimum length, specific content requirements, etc.

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False