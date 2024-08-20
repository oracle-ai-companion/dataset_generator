import os
import aiofiles
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
import re
from src.generator import DatasetGenerator

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
generator = DatasetGenerator()

class MultiTurnRequest(BaseModel):
    prompt: str
    num_turns: int = Field(gt=0)
    model: Literal["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"] = "gemini-pro"
    fine_tuning_format: bool = False
    output_file: Optional[str] = None
    num_conversations: int = Field(gt=0, default=1)

@app.post("/generate/multi-turn")
async def generate_multi_turn(request: MultiTurnRequest, background_tasks: BackgroundTasks):
    async def generate():
        for conversation_index in range(request.num_conversations):
            try:
                full_prompt = f"""Generate a {request.num_turns}-turn dialogue about the following topic. 
                Each turn should start with either 'Human:' or 'AI:' and contain a complete thought or question.
                Topic: {request.prompt}
                
                Human: {request.prompt}"""
                
                logger.debug(f"Full prompt for conversation {conversation_index + 1}: {full_prompt}")
                
                full_dialogue = await generator._generate_content(full_prompt, request.model)
                logger.debug(f"Generated dialogue for conversation {conversation_index + 1}: {full_dialogue}")

                turns = re.split(r'(Human:|AI:)\s*', full_dialogue)
                turns = [turn.strip() for turn in turns if turn.strip()]
                logger.debug(f"Parsed turns for conversation {conversation_index + 1}: {turns}")

                conversation = []
                for i in range(0, len(turns) - 1, 2):
                    speaker = turns[i]
                    content = turns[i + 1]

                    if speaker == "Human:":
                        turn_data = {"input": {"content": content}}
                    elif speaker == "AI:":
                        turn_data = {"output": {"content": content}}
                    else:
                        logger.warning(f"Unexpected speaker in conversation {conversation_index + 1}: {speaker}")
                        continue

                    conversation.append(turn_data)

                try:
                    formatted_data = generator._format_for_fine_tuning({
                        "conversation": conversation,
                        "metadata": {
                            "num_turns": request.num_turns,
                            "model": request.model,
                            "conversation_index": conversation_index
                        }
                    })
                    yield json.dumps(formatted_data) + "\n"
                except Exception as e:
                    logger.error(f"Error in _format_for_fine_tuning for conversation {conversation_index + 1}: {str(e)}")
                    logger.error(f"Conversation data: {conversation}")
                    yield json.dumps({"error": f"Error in formatting conversation {conversation_index + 1}: {str(e)}"}) + "\n"
            except Exception as e:
                logger.error(f"Error during generation of conversation {conversation_index + 1}: {str(e)}")
                yield json.dumps({"error": f"Error in conversation {conversation_index + 1}: {str(e)}"}) + "\n"

    async def save_to_file(content):
        if request.output_file:
            directory = os.path.dirname(request.output_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            async with aiofiles.open(request.output_file, mode='w') as f:
                await f.write(content)
            logger.info(f"Content saved to {request.output_file}")

    if request.output_file:
        content = ""
        async for chunk in generate():
            content += chunk
        background_tasks.add_task(save_to_file, content)
        return JSONResponse(content={"message": f"Generation complete. {request.num_conversations} conversations saved to {request.output_file}"})
    else:
        return StreamingResponse(generate(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)