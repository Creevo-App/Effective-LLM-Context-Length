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
            model="gemini-2.5-flash",
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
def evaluate_aime_dataset_with_padding(dataset_path, max_workers=8):
    """Run Gemini model through AIME 2025 evaluation with different token padding sizes using multiprocessing"""
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset(dataset_path)
    
    # Different token padding sizes to test
    token_sizes = [0, 256, 1024, 4096, 8192, 16384, 32_000, 64_000, 128_000, 256_000, 512_000]
    all_results = {}
    
    total_problems = len(dataset)
    print(f"Evaluating {total_problems} problems with {len(token_sizes)} different token padding sizes...")
    print(f"Using multiprocessing with {max_workers} workers for parallel execution")
    
    for token_size in token_sizes:
        print(f"\n{'='*60}")
        print(f"EVALUATING WITH {token_size} TOKEN PADDING")
        print(f"{'='*60}")
        
        # Prepare arguments for all problems with this token size
        problem_args = []
        for i, item in enumerate(dataset):
            problem_args.append((item, token_size, i, total_problems))
        
        results = []
        correct_count = 0
        completed_count = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all problems for processing
            future_to_problem = {executor.submit(process_problem, args): args for args in problem_args}
            
            try:
                # Process completed futures as they finish
                for future in as_completed(future_to_problem):
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if result['is_correct']:
                        correct_count += 1
                    
                    # Show progress with error indication
                    problem_index = result['problem_index']
                    problem_id = result['problem_id']
                    status_text = "Correct ✓" if result['is_correct'] else "Wrong ✗"
                    if result.get('error'):
                        status_text = "Error ⚠"
                    
                    print(f"Completed {completed_count}/{total_problems} | Problem {problem_index+1} (ID: {problem_id}) | "
                          f"Answer: {result['model_answer']} | {status_text} | "
                          f"Running Accuracy: {correct_count}/{completed_count} ({100*correct_count/completed_count:.1f}%)")
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Evaluation interrupted! Processed {completed_count}/{total_problems} problems.")
                print("Cancelling remaining tasks...")
                # Cancel all pending futures
                for future in future_to_problem:
                    future.cancel()
                # Wait a bit for cancellations to process
                time.sleep(1)
                break
        
        # Sort results by problem index to maintain order
        results.sort(key=lambda x: x['problem_index'])
        
        # Store results for this token size
        accuracy = correct_count / total_problems
        all_results[token_size] = {
            'total_problems': total_problems,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'results': results
        }
        
        print(f"\n=== RESULTS FOR {token_size} TOKEN PADDING ===")
        print(f"Total Problems: {total_problems}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
    
    # Compare results across different token sizes
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS DIFFERENT TOKEN PADDING SIZES")
    print(f"{'='*80}")
    print(f"{'Token Padding':<15} {'Correct':<10} {'Total':<8} {'Accuracy':<12}")
    print("-" * 50)
    
    for token_size in token_sizes:
        result = all_results[token_size]
        print(f"{token_size:<15} {result['correct_count']:<10} {result['total_problems']:<8} {result['accuracy']:<12.2%}")
    
    # Save comprehensive results to file
    output_file = 'aime_2025_token_padding_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_description': 'AIME 2025 evaluation with different token padding sizes (multiprocessed)',
            'token_sizes_tested': token_sizes,
            'max_workers': max_workers,
            'summary_by_token_size': {
                str(k): {
                    'total_problems': v['total_problems'],
                    'correct_count': v['correct_count'],
                    'accuracy': v['accuracy']
                } for k, v in all_results.items()
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    return all_results

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

# Worker function for processing individual problems
def process_problem(args):
    """Process a single problem with given token padding"""
    item, token_padding, problem_index, total_problems = args
    
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
            'error': None
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Failed after retries: {str(e)}"
        logger.error(f"Problem {problem_id} with {token_padding} tokens failed: {error_msg}")
        return {
            'problem_id': problem_id,
            'problem': problem,
            'correct_answer': correct_answer,
            'model_response': f"Error: {error_msg}",
            'model_answer': None,
            'is_correct': False,
            'token_padding': token_padding,
            'problem_index': problem_index,
            'error': error_msg
        }

if __name__ == "__main__":
    try:
        dataset_path = "aime_2025_dataset.json"
        evaluate_aime_dataset_with_padding(dataset_path, max_workers=128)
    except KeyboardInterrupt:
        print("\n🛑 Evaluation stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Evaluation failed with error: {str(e)}")
    finally:
        print("👋 Evaluation session ended.")
