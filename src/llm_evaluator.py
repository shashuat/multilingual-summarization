import json
import torch
import re
import os
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm

def load_gemma_model():
    """Load Gemma 3 27B model with 4-bit quantization"""
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Disable flash attention by setting the environment variable
    os.environ["PYTORCH_ENABLE_FLASH_ATTENTION"] = "0"
    os.environ["PYTORCH_ENABLE_SDPA_FLASH"] = "0"
    
    # Also disable for torch directly
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    model_id = "google/gemma-3-27b-it"
    
    print("Loading Gemma 3 27B processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Loading Gemma 3 27B model with 4-bit quantization...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Force eager implementation instead of flash attention
    )
    model.eval()
    
    return model, processor

def extract_scores_from_response(response):
    """Extract scores from LLM response with multiple fallback methods"""
    # Try to parse JSON first
    try:
        # Find potential JSON objects in the response
        json_pattern = r'\{[\s\S]*?"score_a"[\s\S]*?"score_b"[\s\S]*?\}'
        json_matches = re.findall(json_pattern, response)
        
        if not json_matches:
            json_pattern = r'\{[\s\S]*?"summary_a_score"[\s\S]*?"summary_b_score"[\s\S]*?\}'
            json_matches = re.findall(json_pattern, response)
        
        if json_matches:
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if "score_a" in result and "score_b" in result:
                        return result["score_a"], result["score_b"], result.get("explanation", "")
                    elif "summary_a_score" in result and "summary_b_score" in result:
                        return result["summary_a_score"], result["summary_b_score"], result.get("explanation", "")
                except:
                    continue
        
        # Pattern matching for scores
        score_pattern = r'[Ss]ummary [Aa].*?([1-5]).*?[Ss]ummary [Bb].*?([1-5])'
        score_match = re.search(score_pattern, response)
        
        if score_match:
            return int(score_match.group(1)), int(score_match.group(2)), "Extracted from text"
        
        # Look for any two numbers that could be scores
        numbers = re.findall(r'([0-9]+)', response)
        if len(numbers) >= 2:
            # Filter to only include numbers that could be valid scores (1-5)
            potential_scores = [int(n) for n in numbers if 1 <= int(n) <= 5]
            if len(potential_scores) >= 2:
                return potential_scores[0], potential_scores[1], "Extracted numeric values"
            
        # If we get here, we couldn't extract scores
        return None, None, "Failed to parse response"
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None, f"Error: {str(e)}"

def evaluate_summary_pair(model, processor, title, reference, summary_a, summary_b):
    """Evaluate a pair of summaries using Gemma 3"""
    system_prompt = "You are an expert evaluator of text summaries."
    user_prompt = f"""Your task is to evaluate the quality of two different summaries for the topic: "{title}".

Reference summary: {reference}

Summary A: {summary_a}

Summary B: {summary_b}

Evaluate both summaries on a scale from 1 to 5, where:
- 1 represents the worst possible summary (completely incorrect, misleading, or irrelevant)
- 3 represents an average summary with some good aspects but some issues
- 5 represents a perfect summary (accurate, comprehensive, concise, and well-written)

You should think, reason and Compare the summaries based on:
1. Factual accuracy relative to the reference, 2. Completeness of information, 3. Clarity and readability, 4. Conciseness. Make sure to give different scores for both summaries.

Provide your evaluation in this exact JSON format:
{{
  "score_a": integer from 1 to 5,
  "score_b": integer from 1 to 5,
  "explanation": "A brief explanation of your scores"
}}

Make sure to use only whole numbers (1, 2, 3, 4, or 5) for your scores, not decimals.
"""
    
    # Format using chat template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }
    ]
    
    # Apply chat template and tokenize
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    # Decode response
    response = processor.decode(generation, skip_special_tokens=True)
    
    # Extract scores
    score_a, score_b, explanation = extract_scores_from_response(response)
    
    return {
        "base_score": score_a,
        "finetuned_score": score_b,
        "explanation": explanation,
        "raw_response": response  # Keep for debugging
    }

def main(json_file_path, num_samples=100, output_file="gemma_evaluation_results.json"):
    """Main function to process JSON file and evaluate summaries"""
    # Load JSON data
    print(f"Loading data from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load model and processor
    model, processor = load_gemma_model()
    
    # Get comparisons (limited to specified number)
    comparisons = data.get("comparisons", [])[:num_samples]
    print(f"Evaluating {len(comparisons)} summary pairs...")
    
    results = []
    
    for comparison in tqdm(comparisons, desc="Evaluating summaries"):
        title = comparison.get("title", "")
        reference = comparison.get("reference_summary", "")
        base_summary = comparison.get("base_model_summary", "")
        finetuned_summary = comparison.get("finetuned_model_summary", "")
        
        # Evaluate summaries
        evaluation = evaluate_summary_pair(
            model, processor, title, reference, base_summary, finetuned_summary
        )
        
        # Store results
        result = {
            "index": comparison.get("index"),
            "title": title,
            "base_score": evaluation["base_score"],
            "finetuned_score": evaluation["finetuned_score"],
            "explanation": evaluation["explanation"],
            "improvement": evaluation["finetuned_score"] - evaluation["base_score"] 
                if evaluation["base_score"] is not None and evaluation["finetuned_score"] is not None 
                else None
        }
        
        results.append(result)
        
        # Print progress information
        print(f"\nEvaluated: {title}")
        print(f"  Base score: {evaluation['base_score']}")
        print(f"  Finetuned score: {evaluation['finetuned_score']}")
        if evaluation["base_score"] is not None and evaluation["finetuned_score"] is not None:
            print(f"  Improvement: {evaluation['finetuned_score'] - evaluation['base_score']:.1f}")
    
    # Calculate statistics
    valid_results = [r for r in results if r["base_score"] is not None and r["finetuned_score"] is not None]
    
    if valid_results:
        avg_base_score = sum(r["base_score"] for r in valid_results) / len(valid_results)
        avg_finetuned_score = sum(r["finetuned_score"] for r in valid_results) / len(valid_results)
        avg_improvement = sum(r["improvement"] for r in valid_results) / len(valid_results)
        win_rate = sum(1 for r in valid_results if r["finetuned_score"] > r["base_score"]) / len(valid_results)
        tie_rate = sum(1 for r in valid_results if r["finetuned_score"] == r["base_score"]) / len(valid_results)
        loss_rate = sum(1 for r in valid_results if r["finetuned_score"] < r["base_score"]) / len(valid_results)
        
        statistics = {
            "average_base_score": avg_base_score,
            "average_finetuned_score": avg_finetuned_score,
            "average_improvement": avg_improvement,
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loss_rate": loss_rate,
            "valid_evaluations": len(valid_results),
            "total_evaluations": len(results)
        }
    else:
        statistics = {
            "error": "No valid evaluation results"
        }
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "google/gemma-3-27b-it",
            "config": data.get("config", {}),
            "evaluation_results": results,
            "statistics": statistics
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete. Results saved to {output_file}")
    print(f"Statistics:")
    for key, value in statistics.items():
        print(f"  {key}: {value}")
    
    return results, statistics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate summaries using Gemma 3")
    parser.add_argument("json_file", help="Path to the JSON file containing comparisons")
    parser.add_argument("--samples", type=int, default=100, help="Number of summaries to evaluate (default: 100)")
    parser.add_argument("--output", type=str, default="gemma_evaluation_results.json", 
                        help="Path for the output JSON file (default: gemma_evaluation_results.json)")
    
    args = parser.parse_args()
    main(args.json_file, args.samples, args.output)