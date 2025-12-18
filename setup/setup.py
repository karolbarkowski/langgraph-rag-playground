"""
Setup script that performs all the actions needed to run the langgraph samples.
It performs those actions:
1. Export Nike products from CSV to MongoDB
2. Generate embeddings for products
3. Index and embed policy documents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime
from setup.config import setupConfig
from setup.steps.export_nike_products import export_nike_products
from setup.steps.batch_embed_products import updateProductEmbeddings
from setup.steps.batch_embed_documents import index_documents


def log_step(step_number: int, total_steps: int, step_name: str, status: str = "START"):
    """Log progress for each step"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 70

    if status == "START":
        print(f"\n{separator}")
        print(f"[{timestamp}] STEP {step_number}/{total_steps}: {step_name}")
        print(f"{separator}\n")
    elif status == "COMPLETE":
        print(f"\n{separator}")
        print(f"[{timestamp}] ✓ STEP {step_number}/{total_steps} COMPLETED: {step_name}")
        print(f"{separator}")
    elif status == "ERROR":
        print(f"\n{separator}")
        print(f"[{timestamp}] ✗ STEP {step_number}/{total_steps} FAILED: {step_name}")
        print(f"{separator}")


def main():
    """Run all setup steps sequentially"""
    total_steps = 3
    start_time = time.time()

    print("\n" + "=" * 70)
    print("LANGGRAPH - COMPLETE SETUP PIPELINE")
    print("=" * 70)
    print(f"Starting setup process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        # Step 1: Export Nike products from CSV to MongoDB
        log_step(1, total_steps, "Export Nike Products to MongoDB", "START")
        export_nike_products(setupConfig)
        log_step(1, total_steps, "Export Nike Products to MongoDB", "COMPLETE")

        # Step 2: Generate embeddings for products
        log_step(2, total_steps, "Generate Product Embeddings", "START")
        updateProductEmbeddings(setupConfig)
        log_step(2, total_steps, "Generate Product Embeddings", "COMPLETE")

        # # Step 3: Index and embed policy documents
        log_step(3, total_steps, "Index and Embed Policy Documents", "START")
        index_documents(setupConfig)
        log_step(3, total_steps, "Index and Embed Policy Documents", "COMPLETE")

        # Final summary
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)

        print("\n" + "=" * 70)
        print("SETUP COMPLETE - ALL STEPS SUCCESSFUL")
        print("=" * 70)
        print(f"Total time elapsed: {minutes}m {seconds}s")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\nLangGraph environment is now ready to use!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print("SETUP FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    main()
