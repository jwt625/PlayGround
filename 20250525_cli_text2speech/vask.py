#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    print("🎤 Voice-to-AI Assistant")
    print("Speak your question...")
    
    # Record and transcribe
    try:
        result = subprocess.run(['wspr', '-q'], capture_output=True, text=True, check=True)
        transcript = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error recording: {e}")
        sys.exit(1)
    
    if not transcript:
        print("No speech detected. Please try again.")
        sys.exit(1)
    
    print(f"\n📝 Transcript: {transcript}")
    print("\nOptions:")
    print("  [Enter] - Send to AI")
    print("  [e] - Edit transcript")
    print("  [r] - Record again")
    print("  [q] - Quit")
    
    while True:
        choice = input("\nChoice: ").strip().lower()
        
        if choice == "" or choice == "y":
            # Send to AI
            print(f"\n🤖 AI Response:")
            print("-" * 50)
            try:
                subprocess.run(['chat', transcript], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error getting AI response: {e}")
            break
            
        elif choice == "e":
            # Edit transcript
            print("Enter your edited version:")
            edited_text = input("> ")
            if edited_text.strip():
                transcript = edited_text.strip()
                print(f"\n🤖 AI Response:")
                print("-" * 50)
                try:
                    subprocess.run(['chat', transcript], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error getting AI response: {e}")
                break
            else:
                print("Empty input, keeping original transcript.")
                
        elif choice == "r":
            # Record again
            print("\n🎤 Recording again...")
            try:
                result = subprocess.run(['wspr', '-q'], capture_output=True, text=True, check=True)
                transcript = result.stdout.strip()
                if transcript:
                    print(f"\n📝 New transcript: {transcript}")
                    print("\nOptions:")
                    print("  [Enter] - Send to AI")
                    print("  [e] - Edit transcript")
                    print("  [r] - Record again")
                    print("  [q] - Quit")
                else:
                    print("No speech detected. Please try again.")
            except subprocess.CalledProcessError as e:
                print(f"Error recording: {e}")
                
        elif choice == "q":
            # Quit
            print("Goodbye!")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main() 