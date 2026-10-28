# main.py
from transformers import pipeline
import time

def main():
    print("Loading tiny LLM (distilgpt2)...")
    # 使用极小的模型测试环境
    generator = pipeline('text-generation', model='distilgpt2')
    
    prompt = "Artificial intelligence in research is"
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    result = generator(prompt, max_length=30, num_return_sequences=1)
    end_time = time.time()
    
    print(f"Output: {result[0]['generated_text']}")
    print(f"Inference time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    