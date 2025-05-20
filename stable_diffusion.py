import torch
import os
import csv
import argparse
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion based on prompts from a CSV file")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing prompts")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Directory to save generated images")
    parser.add_argument("--max_images", type=int, default=10000, help="Maximum number of images to generate per dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., bar_1pct)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for generation")
    return parser.parse_args()

def read_prompts_from_csv(csv_path, max_prompts=10000):
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip header if present
        try:
            header = next(reader)
            # Determine which column contains prompts
            prompt_col = 0  # Default to first column, adjust as needed
        except:
            # No header or empty file
            pass
        
        # Read prompts
        for i, row in enumerate(reader):
            if i >= max_prompts:
                break
            if row:  # Skip empty rows
                prompts.append(row[prompt_col])
    
    return prompts

def main():
    args = parse_arguments()
    
    dataset_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)


    # You can use any SD model you want. Our paper used SD2.5 previously back in 2023.    
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    
    # Read prompts from CSV
    print(f"Reading prompts from {args.csv_path}...")
    prompts = read_prompts_from_csv(args.csv_path, args.max_images)
    
    # Generate images
    print(f"Generating {len(prompts)} images for dataset {args.dataset_name}...")
    for i, prompt in enumerate(tqdm(prompts)):
        # Generate image
        image = pipe(
            prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        
        # Save image
        image_path = os.path.join(dataset_dir, f"{args.dataset_name}_{i:05d}.png")
        image.save(image_path)
        
    print(f"Generation complete. {len(prompts)} images saved in {dataset_dir}")

if __name__ == "__main__":
    main()
    
# command
# python stable_diffusion.py --csv_path path/to/filter_csv_file.csv --dataset_name bar_1pct --max_images 10000