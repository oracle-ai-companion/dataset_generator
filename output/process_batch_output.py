import asyncio
import aiofiles
import json
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def process_line(line):
    try:
        data = json.loads(line)
        # Process the data as needed
        # For now, we're just returning the data as-is
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON from line: {line}")
        return None

async def process_file(filename):
    async with aiofiles.open(filename, mode='r') as f:
        tasks = []
        async for line in f:
            task = asyncio.create_task(process_line(line))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

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
    
    with ThreadPoolExecutor() as executor:
        executor.submit(write_output, results, output_file)

if __name__ == '__main__':
    asyncio.run(main())