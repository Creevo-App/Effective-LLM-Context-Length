import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')

# Load AIME 2025 dataset
def load_dataset(file_path):
    """Load the AIME 2025 dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Create prompt for mathematical problem solving
def create_math_prompt(problem):
    """Create a structured prompt for solving AIME problems"""
    prompt = f"""You are solving an AIME (American Invitational Mathematics Examination) problem. 
AIME problems require integer answers between 0 and 999.

Problem: {problem}

Please solve this step by step and provide your final answer as a single integer between 0 and 999.
Format your response with "Final Answer: [your integer answer]" at the end."""
    return prompt

# Evaluate model on AIME dataset
def evaluate_aime_dataset(dataset_path):
    """Run Gemini model through AIME 2025 evaluation"""
    print("Loading AIME 2025 dataset...")
    dataset = load_dataset(dataset_path)
    
    results = []
    correct_count = 0
    total_problems = len(dataset)
    
    print(f"Evaluating {total_problems} problems...")
    
    for i, item in enumerate(dataset):
        problem_id = item['id']
        problem = item['problem']
        correct_answer = item['answer']
        
        print(f"\n--- Problem {i+1}/{total_problems} (ID: {problem_id}) ---")
        print(f"Problem: {problem[:100]}..." if len(problem) > 100 else f"Problem: {problem}")
        
        # Create prompt and get model response
        prompt = create_math_prompt(problem)
        
        try:
            response = model.generate_content(prompt)
            model_response = response.text
            
            # Extract answer from response
            model_answer = extract_answer(model_response)
            
            # Check if correct
            is_correct = str(model_answer) == str(correct_answer)
            if is_correct:
                correct_count += 1
            
            # Store result
            result = {
                'problem_id': problem_id,
                'problem': problem,
                'correct_answer': correct_answer,
                'model_response': model_response,
                'model_answer': model_answer,
                'is_correct': is_correct
            }
            results.append(result)
            
            print(f"Correct Answer: {correct_answer}")
            print(f"Model Answer: {model_answer}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print(f"Current Accuracy: {correct_count}/{i+1} ({100*correct_count/(i+1):.1f}%)")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {e}")
            result = {
                'problem_id': problem_id,
                'problem': problem,
                'correct_answer': correct_answer,
                'model_response': f"Error: {e}",
                'model_answer': None,
                'is_correct': False
            }
            results.append(result)
    
    # Final results
    accuracy = correct_count / total_problems
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Problems: {total_problems}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results to file
    output_file = 'aime_2025_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_problems': total_problems,
                'correct_count': correct_count,
                'accuracy': accuracy
            },
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

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

if __name__ == "__main__":
    dataset_path = "aime_2025_dataset.json"
    evaluate_aime_dataset(dataset_path)
