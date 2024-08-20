import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json
import re
from src.generator import DatasetGenerator
from io import StringIO
import tempfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dataset Generator API",
    description="API for generating multi-turn conversational datasets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

generator = DatasetGenerator()

class MultiTurnRequest(BaseModel):
    prompt: str
    num_turns: int = Field(gt=0)
    model: Literal["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"] = "gemini-pro"
    fine_tuning_format: bool = False
    output_file: Optional[str] = None
    num_conversations: int = Field(gt=0, default=1)

@app.get("/")
async def root():
    return {"message": "Welcome to the Dataset Generator API"}

@app.post("/generate/multi-turn")
async def generate_multi_turn(request: MultiTurnRequest):
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

    content = StringIO()
    async for chunk in generate():
        content.write(chunk)
    
    content.seek(0)
    
    if request.output_file:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl') as temp_file:
            temp_file.write(content.getvalue())
            temp_file_path = temp_file.name

        # Return the file as a downloadable response
        return FileResponse(
            temp_file_path,
            media_type="application/json",
            filename=request.output_file
        )
    else:
        return StreamingResponse(content, media_type="application/json")

@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "The requested resource was not found"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)