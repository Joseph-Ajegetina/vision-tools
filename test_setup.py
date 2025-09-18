#!/usr/bin/env python3
"""
Test script for automatic CINIC-10 setup.

This script demonstrates how the automatic setup works,
similar to the landmark identifier project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("🧪 Testing CINIC-10 Automatic Setup")
    print("=" * 50)

    try:
        # Import and run setup
        from src.utils.setup import setup_env, verify_setup

        print("🚀 Running setup_env()...")
        setup_info = setup_env()

        print(f"\n📊 Setup Results:")
        print(f"   Device: {setup_info['device']}")
        print(f"   CUDA Available: {setup_info['cuda_available']}")
        print(f"   Random Seed: {setup_info['seed']}")

        print(f"\n🔍 Verifying setup...")
        if verify_setup():
            print("✅ All checks passed!")

            # Test data loading
            print(f"\n🧪 Testing data loading...")
            from src.data.dataset import CINIC10DataModule

            config = setup_info['config']
            data_module = CINIC10DataModule(
                data_dir=config['dataset']['data_dir'],
                batch_size=16,  # Small batch for testing
                num_workers=2
            )

            data_loaders = data_module.setup_data_loaders()

            # Test getting a batch
            train_batch = next(iter(data_loaders['train']))
            print(f"✅ Successfully loaded batch: {train_batch[0].shape}")

            dataset_info = data_module.get_dataset_info()
            print(f"✅ Dataset info: {dataset_info['num_classes']} classes, {sum(dataset_info.get('train_samples', [0]))} training samples")

        else:
            print("❌ Setup verification failed")
            return False

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n🎉 Setup test completed successfully!")
    print(f"💡 You can now run the notebook: notebooks/comprehensive_walkthrough.ipynb")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)