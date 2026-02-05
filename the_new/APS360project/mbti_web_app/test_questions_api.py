"""
Test script for Questions API endpoints
Run this after starting the Flask server to test the new API
"""

import requests
import json

BASE_URL = "http://localhost:5000"


def test_get_questions(language='en', count=3):
    """Test GET /api/questions endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing GET /api/questions (language={language}, count={count})")
    print('='*60)

    url = f"{BASE_URL}/api/questions"
    params = {
        'language': language,
        'count': count,
        'balanced': 'true'
    }

    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"Session ID: {data['session_id']}")
            print(f"Language: {data['language']}")
            print(f"Version: {data['version']}")
            print(f"Question Count: {data['count']}")
            print(f"\nQuestions:")
            for i, q in enumerate(data['questions'], 1):
                print(f"\n  {i}. [{q['dimension']}] {q['question']}")
                print(f"     {q['description'][:80]}...")
            return data
        else:
            print(f"❌ Error: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def test_get_question_by_id(question_id=1, language='en'):
    """Test GET /api/questions/<id> endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing GET /api/questions/{question_id} (language={language})")
    print('='*60)

    url = f"{BASE_URL}/api/questions/{question_id}"
    params = {'language': language}

    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"ID: {data['id']}")
            print(f"Question: {data['question']}")
            print(f"Dimension: {data['dimension']}")
            print(f"Description: {data['description']}")
            return data
        else:
            print(f"❌ Error: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def test_record_question_usage():
    """Test POST /api/questions/usage endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing POST /api/questions/usage")
    print('='*60)

    url = f"{BASE_URL}/api/questions/usage"
    payload = {
        "prediction_id": "test-prediction-12345",
        "session_id": "test-session-67890",
        "questions": [
            {
                "question_id": 1,
                "question_order": 1,
                "user_response": "I prefer deep conversations with a few close friends."
            },
            {
                "question_id": 2,
                "question_order": 2,
                "user_response": "I like to explore new possibilities and innovative approaches."
            },
            {
                "question_id": 3,
                "question_order": 3,
                "user_response": "I prefer to stay flexible and adjust plans as needed."
            }
        ]
    }

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"Status: {data['status']}")
            print(f"Recorded: {data['recorded']} questions")
            print(f"Note: {data.get('note', 'N/A')}")
            return data
        else:
            print(f"❌ Error: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def test_get_question_stats(language='en'):
    """Test GET /api/questions/stats endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing GET /api/questions/stats (language={language})")
    print('='*60)

    url = f"{BASE_URL}/api/questions/stats"
    params = {'language': language}

    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"Total Questions: {data['total_questions']}")
            print(f"Language: {data['language']}")
            print(f"\nQuestions by Dimension:")
            for dim, count in data['by_dimension'].items():
                print(f"  {dim}: {count} questions")
            return data
        else:
            print(f"❌ Error: {response.json()}")
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MBTI Question Management API - Test Suite")
    print("="*60)
    print("\nMake sure the Flask server is running on http://localhost:5000")
    input("\nPress Enter to start tests...")

    # Test 1: Get random questions (English)
    test_get_questions(language='en', count=3)

    # Test 2: Get random questions (Chinese)
    test_get_questions(language='zh', count=3)

    # Test 3: Get specific question by ID
    test_get_question_by_id(question_id=1, language='en')
    test_get_question_by_id(question_id=1, language='zh')

    # Test 4: Record question usage
    test_record_question_usage()

    # Test 5: Get question statistics
    test_get_question_stats(language='en')
    test_get_question_stats(language='zh')

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
