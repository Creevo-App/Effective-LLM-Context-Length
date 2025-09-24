import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for clients to avoid sharing across threads
thread_local = threading.local()

# Configuration parameters
context = 'math'  # 'random' or 'math'
model = "gemini-2.5-flash"

def get_client():
    """Get a thread-local client instance"""
    if not hasattr(thread_local, 'client'):
        # The new API doesn't need explicit configuration, it uses environment variables
        thread_local.client = genai.Client(api_key=GEMINI_API_KEY)
    return thread_local.client

# Global cache for math content and cached content objects
math_content_cache = {}
cached_content_objects = {}

def load_math_content(num_words):
    """Load math content from AOPS_Raw_Text.txt file"""
    if num_words == 0:
        return ""
    
    # Check cache first
    if num_words in math_content_cache:
        return math_content_cache[num_words]
    
    try:
        with open('AOPS_Raw_Text.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into words and take the first num_words
        words = content.split()
        if len(words) >= num_words:
            selected_words = words[:num_words]
        else:
            # If we need more words than available, repeat the content
            repetitions = (num_words // len(words)) + 1
            extended_words = (words * repetitions)[:num_words]
            selected_words = extended_words
        
        math_text = " ".join(selected_words)
        
        # Cache the result
        math_content_cache[num_words] = math_text
        return math_text
        
    except FileNotFoundError:
        logger.error("AOPS_Raw_Text.txt file not found. Falling back to random words.")
        return generate_random_words(num_words)
    except Exception as e:
        logger.error(f"Error loading math content: {e}. Falling back to random words.")
        return generate_random_words(num_words)

# Define retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def make_api_call_with_retry(client, prompt, cached_content_name=None):
    """Make API call with retry logic and optional caching"""
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024)
        )
        
        # Add cached content if provided
        if cached_content_name:
            config.cached_content = cached_content_name
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        logger.warning(f"API call failed: {str(e)}. Retrying...")
        raise

def create_all_caches_upfront(token_sizes):
    """Create all cached content upfront to reuse across workers"""
    if context != 'math':
        return
        
    logger.info("Creating cached content for math context...")
    client = get_client()
    
    for token_padding in token_sizes:
        if token_padding >= 1024:  # Only cache for sizes >= 1024 tokens (API requirement)
            cache_key = f"{context}_{token_padding}"
            
            # Skip if already cached
            if cache_key in cached_content_objects:
                continue
                
            try:
                context_text = load_math_content(token_padding)
                
                # Create cache with math content
                cache = client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        display_name=f'math_context_{token_padding}_tokens',
                        system_instruction=(
                            'You are solving AIME (American Invitational Mathematics Examination) problems. '
                            'AIME problems require integer answers between 0 and 999. '
                            'The following mathematical content provides context for your reasoning.'
                        ),
                        contents=[
                            types.Content(
                                role="user", 
                                parts=[types.Part(text=context_text)]
                            )
                        ],
                        ttl="3600s",  # 1 hour TTL
                    )
                )
                
                cached_content_objects[cache_key] = cache.name
                logger.info(f"Created cached content for {token_padding} tokens: {cache.name}")
                
            except Exception as e:
                logger.error(f"Failed to create cached content for {token_padding} tokens: {e}")
    
    logger.info(f"Cached content creation complete. Created {len(cached_content_objects)} caches.")

# Load AIME 2025 dataset
def load_dataset(file_path):
    """Load the AIME 2025 dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Generate context content based on context type
def generate_context_content(num_words):
    """Generate context content based on the global context setting"""
    if num_words == 0:
        return ""
    
    if context == 'math':
        return load_math_content(num_words)
    else:  # context == 'random'
        return generate_random_words(num_words)

# Generate random words for padding
def generate_random_words(num_words):
    """Generate random words to add as padding to prompts"""
    if num_words == 0:
        return ""
    
    # Simple word list for generating random padding
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "she", "or", "an",
        "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
        "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after",
        "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "is", "water", "long", "find", "here", "thing",
        "great", "man", "world", "life", "still", "hand", "high", "right", "small", "large", "next",
        "early", "group", "important", "begin", "seem", "country", "help", "talk", "where", "turn",
        "problem", "every", "start", "thought", "study", "night", "move", "live", "Mr", "point",
        "believe", "hold", "today", "bring", "happen", "without", "before", "large", "million", "must",
        "home", "under", "water", "room", "write", "mother", "area", "national", "money", "story",
        "young", "fact", "month", "different", "lot", "right", "study", "book", "eye", "job",
        "word", "though", "business", "issue", "side", "kind", "four", "head", "far", "black",
        "long", "both", "little", "house", "yes", "after", "since", "long", "provide", "service"
    ]
    
    # Generate random words
    random_words = []
    for _ in range(num_words):
        random_words.append(random.choice(common_words))
    
    return " ".join(random_words)

# Create prompt for mathematical problem solving
def create_math_prompt(problem, token_padding=0):
    """Create a structured prompt for solving AIME problems with optional context"""
    if context == 'math' and token_padding >= 1024:
        # For math context with caching (>=1024 tokens), just return the problem
        prompt = f"""Problem: {problem}

Please solve this step by step and provide your final answer as a single integer between 0 and 999.
Format your response with "Final Answer: [your integer answer]" at the end."""
    else:
        # For random context, no padding, or small math context (<1024 tokens), include context directly in prompt
        padding = ""
        if token_padding > 0:
            padding = generate_context_content(token_padding) + "\n\n"
        
        prompt = f"""{padding}You are solving an AIME (American Invitational Mathematics Examination) problem. 
AIME problems require integer answers between 0 and 999.

Problem: {problem}

Please solve this step by step and provide your final answer as a single integer between 0 and 999.
Format your response with "Final Answer: [your integer answer]" at the end."""
    
    return prompt

# Evaluate model on AIME dataset with different token padding sizes using batch processing
def evaluate_aime_dataset_with_padding(dataset_path, num_runs=5, max_workers=8):
    """Run Gemini model through AIME 2025 evaluation with different token padding sizes using batch processing"""
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset(dataset_path)
    
    # Different token padding sizes to test (limited to 128k)
    token_sizes = [0, 256, 1024, 4096, 8192, 16384, 32_000, 64_000, 128_000]
    
    total_problems = len(dataset)
    total_tasks = total_problems * len(token_sizes) * num_runs
    print(f"Evaluating {total_problems} problems with {len(token_sizes)} different token padding sizes...")
    print(f"Running {num_runs} times per problem-token combination = {total_tasks} total evaluations")
    print(f"Processing in batches for efficiency")
    
    # Create all caches upfront to avoid duplicate cache creation
    create_all_caches_upfront(token_sizes)
    
    print(f"\n{'='*80}")
    print("STARTING BATCH EVALUATION OF ALL COMBINATIONS")
    print(f"{'='*80}")
    
    # Process everything as batches by token size
    raw_results = []
    completed_count = 0
    
    for token_size in token_sizes:
        print(f"\nProcessing token size: {token_size}")
        
        # Create batch of all problems for this token size
        batch_prompts = []
        batch_metadata = []
        
        for problem_index, item in enumerate(dataset):
            for run_number in range(1, num_runs + 1):
                prompt = create_math_prompt(item['problem'], token_size)
                batch_prompts.append(prompt)
                batch_metadata.append({
                    'item': item,
                    'token_padding': token_size,
                    'problem_index': problem_index,
                    'run_number': run_number
                })
        
        # Process this batch
        batch_results = process_batch(batch_prompts, batch_metadata, token_size)
        raw_results.extend(batch_results)
        completed_count += len(batch_results)
        
        print(f"Completed {completed_count}/{total_tasks} | Progress: {100*completed_count/total_tasks:.1f}%")
    
    # Aggregate results by token size
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS BY TOKEN SIZE")
    print(f"{'='*80}")
    
    aggregated_results = {}
    for token_size in token_sizes:
        # Get all results for this token size
        token_results = [r for r in raw_results if r['token_padding'] == token_size]
        
        # Group by problem
        problem_results = {}
        for result in token_results:
            problem_id = result['problem_id']
            if problem_id not in problem_results:
                problem_results[problem_id] = []
            problem_results[problem_id].append(result)
        
        # Calculate average accuracy per problem
        problem_accuracies = []
        for problem_id, results in problem_results.items():
            correct_runs = sum(1 for r in results if r['is_correct'])
            total_runs = len(results)
            if total_runs > 0:
                problem_accuracy = correct_runs / total_runs
                problem_accuracies.append(problem_accuracy)
        
        # Calculate overall statistics
        if problem_accuracies:
            mean_accuracy = sum(problem_accuracies) / len(problem_accuracies)
            std_accuracy = (sum((acc - mean_accuracy) ** 2 for acc in problem_accuracies) / len(problem_accuracies)) ** 0.5
            std_error = std_accuracy / (len(problem_accuracies) ** 0.5)
        else:
            mean_accuracy = std_accuracy = std_error = 0.0
        
        aggregated_results[token_size] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'std_error': std_error,
            'total_problems': len(problem_accuracies),
            'total_runs': len(token_results),
            'problem_accuracies': problem_accuracies,
            'raw_results': token_results
        }
        
        print(f"Token Size {token_size}: {mean_accuracy:.2%} ± {std_error:.2%} (n={len(problem_accuracies)} problems)")
    
    # Display comparison
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS DIFFERENT TOKEN PADDING SIZES")
    print(f"{'='*80}")
    print(f"{'Token Padding':<15} {'Mean Accuracy':<15} {'Std Error':<12} {'Problems':<10}")
    print("-" * 60)
    
    for token_size in token_sizes:
        result = aggregated_results[token_size]
        print(f"{token_size:<15} {result['mean_accuracy']:<15.2%} {result['std_error']:<12.2%} {result['total_problems']:<10}")
    
    # Save comprehensive results to file
    output_dir = f'{model}_{context}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/aime_2025_token_padding_evaluation_results.json'
    
    # Ensure the output directory exists and is writable
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions by creating a temporary file
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Cannot create or write to output directory '{output_dir}': {e}")
        # Fallback to current directory
        output_dir = '.'
        output_file = f'aime_2025_token_padding_evaluation_results_{model}_{context}.json'
        logger.info(f"Falling back to current directory. Output file: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_description': f'AIME 2025 evaluation with {context} context and different token padding sizes ({num_runs} runs per problem-token combination)',
            'model': model,
            'context_type': context,
            'token_sizes_tested': token_sizes,
            'num_runs': num_runs,
            'max_workers': max_workers,
            'total_evaluations': len(raw_results),
            'caching_enabled': context == 'math',
            'summary_by_token_size': {
                str(k): {
                    'mean_accuracy': v['mean_accuracy'],
                    'std_accuracy': v['std_accuracy'],
                    'std_error': v['std_error'],
                    'total_problems': v['total_problems'],
                    'total_runs': v['total_runs']
                } for k, v in aggregated_results.items()
            },
            'raw_results': raw_results,
            'aggregated_results': aggregated_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    return aggregated_results

def extract_answer(response_text):
    """Extract the final answer from model response"""
    # Look for "Final Answer: X" pattern
    import re
    
    # Try to find "Final Answer: NUMBER" pattern
    final_answer_match = re.search(r'Final Answer:\s*(\d+)', response_text, re.IGNORECASE)
    if final_answer_match:
        return int(final_answer_match.group(1))
    
    # Fallback: look for last number in the response
    numbers = re.findall(r'\b\d{1,3}\b', response_text)
    if numbers:
        return int(numbers[-1])
    
    # If no number found, return None
    return None

# Batch processing function
def process_batch(batch_prompts, batch_metadata, token_size):
    """Process a batch of prompts for a given token size"""
    results = []
    
    # Get cached content for this token size if applicable
    cached_content_name = None
    if context == 'math' and token_size >= 1024:
        cache_key = f"{context}_{token_size}"
        cached_content_name = cached_content_objects.get(cache_key)
    
    # Get client
    client = get_client()
    
    print(f"  Processing {len(batch_prompts)} prompts for token size {token_size}...")
    
    # Process all prompts in this batch
    for i, (prompt, metadata) in enumerate(zip(batch_prompts, batch_metadata)):
        try:
            item = metadata['item']
            problem_id = item['id']
            problem = item['problem']
            correct_answer = item['answer']
            token_padding = metadata['token_padding']
            problem_index = metadata['problem_index']
            run_number = metadata['run_number']
            
            # Get model response with retry logic
            model_response = make_api_call_with_retry(client, prompt, cached_content_name)
            
            # Extract answer
            extracted_answer = extract_answer(model_response)
            is_correct = str(extracted_answer) == str(correct_answer)
            
            result = {
                'problem_id': problem_id,
                'problem': problem,
                'correct_answer': correct_answer,
                'model_response': model_response,
                'model_answer': extracted_answer,
                'is_correct': is_correct,
                'token_padding': token_padding,
                'problem_index': problem_index,
                'run_number': run_number,
                'context': context,
                'cached_content_used': cached_content_name is not None,
                'error': None
            }
            
            results.append(result)
            
            # Show progress within batch
            if (i + 1) % 10 == 0 or (i + 1) == len(batch_prompts):
                status_text = "✓" if result['is_correct'] else "✗"
                print(f"    Batch progress: {i+1}/{len(batch_prompts)} | Problem {problem_index+1} Run {run_number} | {status_text}")
            
        except Exception as e:
            error_msg = f"Failed after retries: {str(e)}"
            logger.error(f"Problem {problem_id} with {token_size} tokens, run {run_number} failed: {error_msg}")
            
            results.append({
                'problem_id': item['id'],
                'problem': item['problem'],
                'correct_answer': item['answer'],
                'model_response': f"Error: {error_msg}",
                'model_answer': None,
                'is_correct': False,
                'token_padding': token_padding,
                'problem_index': metadata['problem_index'],
                'run_number': metadata['run_number'],
                'context': context,
                'cached_content_used': False,
                'error': error_msg
            })
    
    print(f"  Completed batch for token size {token_size}")
    return results

if __name__ == "__main__":
    try:
        dataset_path = "aime_2025_dataset.json"
        evaluate_aime_dataset_with_padding(dataset_path, num_runs=10, max_workers=16)
    except KeyboardInterrupt:
        print("\n🛑 Evaluation stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Evaluation failed with error: {str(e)}")
    finally:
        print("👋 Evaluation session ended.")
