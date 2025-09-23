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

model = "gemini-2.5-pro"

def get_client():
    """Get a thread-local client instance"""
    if not hasattr(thread_local, 'client'):
        # The new API doesn't need explicit configuration, it uses environment variables
        thread_local.client = genai.Client(api_key=GEMINI_API_KEY)
    return thread_local.client

# Define retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def make_api_call_with_retry(client, prompt):
    """Make API call with retry logic"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            )
        )
        return response.text
    except Exception as e:
        logger.warning(f"API call failed: {str(e)}. Retrying...")
        raise

# Load AIME 2025 dataset
def load_dataset(file_path):
    """Load the AIME 2025 dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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
    """Create a structured prompt for solving AIME problems with optional padding"""
    # Add random word padding if specified
    padding = ""
    if token_padding > 0:
        padding = generate_random_words(token_padding) + "\n\n"
    
    prompt = f"""{padding}You are solving an AIME (American Invitational Mathematics Examination) problem. 
AIME problems require integer answers between 0 and 999.

Problem: {problem}

Please solve this step by step and provide your final answer as a single integer between 0 and 999.
Format your response with "Final Answer: [your integer answer]" at the end."""
    return prompt

# Evaluate model on AIME dataset with different token padding sizes using multiprocessing
def evaluate_aime_dataset_with_padding(dataset_path, num_runs=5, max_workers=8):
    """Run Gemini model through AIME 2025 evaluation with different token padding sizes using multiprocessing"""
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset(dataset_path)
    
    # Different token padding sizes to test
    token_sizes = [0, 256, 1024, 4096, 8192, 16384, 32_000, 64_000, 128_000, 256_000]
    
    total_problems = len(dataset)
    total_tasks = total_problems * len(token_sizes) * num_runs
    print(f"Evaluating {total_problems} problems with {len(token_sizes)} different token padding sizes...")
    print(f"Running {num_runs} times per problem-token combination = {total_tasks} total evaluations")
    print(f"Using multiprocessing with {max_workers} workers for parallel execution")
    
    # Prepare all arguments for all problem-token-run combinations
    all_args = []
    for problem_index, item in enumerate(dataset):
        for token_size in token_sizes:
            for run_number in range(1, num_runs + 1):
                all_args.append((item, token_size, problem_index, run_number))
    
    # Track results
    raw_results = []
    completed_count = 0
    
    print(f"\n{'='*80}")
    print("STARTING PARALLEL EVALUATION OF ALL COMBINATIONS")
    print(f"{'='*80}")
    
    # Use ThreadPoolExecutor for parallel processing of ALL combinations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all problem-token-run combinations for processing
        future_to_args = {executor.submit(process_problem_run, args): args for args in all_args}
        
        try:
            # Process completed futures as they finish
            for future in as_completed(future_to_args):
                result = future.result()
                raw_results.append(result)
                completed_count += 1
                
                # Show progress
                problem_index = result['problem_index']
                problem_id = result['problem_id']
                token_padding = result['token_padding']
                run_number = result['run_number']
                status_text = "✓" if result['is_correct'] else "✗"
                if result.get('error'):
                    status_text = "⚠"
                
                print(f"Completed {completed_count}/{total_tasks} | Problem {problem_index+1} | "
                      f"Tokens: {token_padding} | Run: {run_number} | {status_text} | "
                      f"Progress: {100*completed_count/total_tasks:.1f}%")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Evaluation interrupted! Processed {completed_count}/{total_tasks} tasks.")
            print("Cancelling remaining tasks...")
            # Cancel all pending futures
            for future in future_to_args:
                future.cancel()
            # Wait a bit for cancellations to process
            time.sleep(1)
    
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
    output_file = f'{model}/aime_2025_token_padding_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_description': f'AIME 2025 evaluation with different token padding sizes ({num_runs} runs per problem-token combination)',
            'token_sizes_tested': token_sizes,
            'num_runs': num_runs,
            'max_workers': max_workers,
            'total_evaluations': len(raw_results),
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

# Worker function for processing individual problem-token-run combinations
def process_problem_run(args):
    """Process a single problem with given token padding for a specific run"""
    item, token_padding, problem_index, run_number = args
    
    problem_id = item['id']
    problem = item['problem']
    correct_answer = item['answer']
    
    # Get thread-local client
    client = get_client()
    
    try:
        # Create prompt and get model response with retry logic
        prompt = create_math_prompt(problem, token_padding)
        model_response = make_api_call_with_retry(client, prompt)
        
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
            'error': None
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Failed after retries: {str(e)}"
        logger.error(f"Problem {problem_id} with {token_padding} tokens, run {run_number} failed: {error_msg}")
        return {
            'problem_id': problem_id,
            'problem': problem,
            'correct_answer': correct_answer,
            'model_response': f"Error: {error_msg}",
            'model_answer': None,
            'is_correct': False,
            'token_padding': token_padding,
            'problem_index': problem_index,
            'run_number': run_number,
            'error': error_msg
        }

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
