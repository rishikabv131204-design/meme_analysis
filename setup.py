#!/usr/bin/env python3
"""
Setup script for the Meme Interpretation Pipeline
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Google API Key for Gemini\n")
            f.write("# Get your API key from: https://makersuite.google.com/app/apikey\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
        print("✅ Created .env file. Please add your Google API key.")
    else:
        print("✅ .env file already exists.")

def main():
    print("🚀 Setting up Meme Interpretation Pipeline...")
    
    if install_requirements():
        create_env_file()
        print("\n📋 Next steps:")
        print("1. Get your Google API key from: https://makersuite.google.com/app/apikey")
        print("2. Edit the .env file and add your API key")
        print("3. Run: python v.py")
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
