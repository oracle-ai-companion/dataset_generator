import asyncio
import aiohttp
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def process_instance(session, instance):
    input_content = instance['input']['content']
    url = "https://api.example.com/generate"  # Replace with your actual API endpoint
    
    async with session.post(url, json={"prompt": input_content}) as response:
        if response.status == 200:
            result = await response.json()
            instance['output'] = result
        else:
            logging.error(f"Error processing instance: {response.status}")
            instance['output'] = {"error": f"HTTP {response.status}"}
    
    return instance

async def process_model_instances(model_data):
    async with aiohttp.ClientSession() as session:
        tasks = [process_instance(session, instance) for instance in model_data['instances']]
        processed_instances = await asyncio.gather(*tasks)
    
    model_data['instances'] = processed_instances
    return model_data

def process_model(model_data):
    return asyncio.run(process_model_instances(model_data))

async def process_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, partial(process_model, model_data)) 
                 for model_data in data]
        processed_data = await asyncio.gather(*tasks)
    
    return processed_data

def write_output(data, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Output written to {output_file}")
    except IOError:
        logging.error(f"Failed to write output to {output_file}")

async def main():
    input_file = 'output/batch_output.jsonl'
    output_file = 'output/processed_output.json'
    
    logging.info(f"Starting to process {input_file}")
    results = await process_file(input_file)
    logging.info(f"Finished processing {input_file}")
    
    write_output(results, output_file)

if __name__ == '__main__':
    asyncio.run(main())