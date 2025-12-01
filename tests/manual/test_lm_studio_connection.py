"""
Quick connection test for LM Studio.

This script checks if LM Studio is running and accessible.

Usage:
    python tests/manual/test_lm_studio_connection.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import httpx


def test_lm_studio_connection():
    """Test if LM Studio server is accessible."""
    print("=" * 60)
    print("LM Studio Connection Test")
    print("=" * 60)
    
    base_url = "http://localhost:1234"
    
    print(f"\nAttempting to connect to: {base_url}")
    print("Timeout: 5 seconds\n")
    
    try:
        # Try to connect to the server
        client = httpx.Client(timeout=5.0)
        response = client.get(f"{base_url}/v1/models")
        
        if response.status_code == 200:
            print("✓ Successfully connected to LM Studio!")
            print(f"  Status Code: {response.status_code}")
            
            # Try to parse the response
            try:
                data = response.json()
                if "data" in data:
                    models = data["data"]
                    print(f"  Available models: {len(models)}")
                    for model in models:
                        print(f"    - {model.get('id', 'Unknown')}")
                else:
                    print("  Response:", data)
            except Exception:
                print("  Response:", response.text[:200])
            
            print("\n✓ LM Studio is ready to use!")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except httpx.ConnectError:
        print("✗ Connection failed: Could not connect to LM Studio")
        print("\nTroubleshooting:")
        print("1. Is LM Studio installed and running?")
        print("2. Is the local server started in LM Studio?")
        print("   (Look for 'Start Server' button in LM Studio)")
        print("3. Is the server running on the default port (1234)?")
        print(f"   (Check LM Studio settings)")
        return False
        
    except httpx.TimeoutException:
        print("✗ Connection timeout: LM Studio is not responding")
        print("\nTroubleshooting:")
        print("1. Is a model loaded in LM Studio?")
        print("2. Is the server actually started?")
        print("3. Try restarting LM Studio")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    finally:
        try:
            client.close()
        except:
            pass


def test_simple_request():
    """Test a simple completion request."""
    print("\n" + "=" * 60)
    print("Simple Completion Test")
    print("=" * 60)
    
    try:
        from src.rag.llm_interface import LLMInterface, LLMError
        
        print("\nInitializing LLM interface...")
        llm = LLMInterface()
        
        print("Sending test request...")
        print("(This may take 10-30 seconds depending on your model)\n")
        
        question = "What is 2+2?"
        context = ["Basic arithmetic: 2+2 equals 4."]
        
        answer = llm.generate(question, context)
        
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
        print("✓ Successfully generated response!")
        return True
        
    except LLMError as e:
        print(f"✗ LLM Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("\nStep 1: Testing Connection")
    connection_ok = test_lm_studio_connection()
    
    if connection_ok:
        print("\nStep 2: Testing Simple Request")
        user_input = input("\nDo you want to test a simple completion? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            test_simple_request()
        else:
            print("\nSkipping completion test.")
    else:
        print("\n⚠️  Cannot proceed with completion test until connection is established.")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
