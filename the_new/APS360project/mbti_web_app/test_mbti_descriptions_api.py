"""
测试MBTI描述API端点

运行方式:
1. 启动服务器: python app.py
2. 在另一个终端运行: python test_mbti_descriptions_api.py
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def print_section(title):
    """打印美观的分隔线"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_get_all_descriptions_en():
    """测试获取所有MBTI描述（英文）"""
    print_section("Test 1: GET /api/mbti/descriptions?language=en")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions?language=en")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"   Language: {data.get('language')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Types loaded: {len(data.get('types', {}))}")
            print(f"   Type names: {', '.join(data.get('types', {}).keys())}")

            # Show one example
            if 'types' in data and 'INTJ' in data['types']:
                intj = data['types']['INTJ']
                print(f"\n   Example (INTJ):")
                print(f"     Title: {intj.get('title')}")
                print(f"     Description: {intj.get('description')[:100]}...")
                print(f"     Characteristics: {', '.join(intj.get('characteristics', [])[:3])}...")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_get_all_descriptions_zh():
    """测试获取所有MBTI描述（中文）"""
    print_section("Test 2: GET /api/mbti/descriptions?language=zh")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions?language=zh")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"   Language: {data.get('language')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Types loaded: {len(data.get('types', {}))}")

            # Show one example
            if 'types' in data and 'INTJ' in data['types']:
                intj = data['types']['INTJ']
                print(f"\n   Example (INTJ):")
                print(f"     Title: {intj.get('title')}")
                print(f"     Description: {intj.get('description')[:80]}...")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_get_specific_type_en():
    """测试获取特定MBTI类型描述（英文）"""
    print_section("Test 3: GET /api/mbti/descriptions/INTJ?language=en")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions/INTJ?language=en")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"   Type: {data.get('type')}")
            print(f"   Title: {data.get('title')}")
            print(f"   Description: {data.get('description')}")
            print(f"   Characteristics: {', '.join(data.get('characteristics', []))}")
            print(f"   Career Paths: {data.get('career_paths')}")
            print(f"   Development Suggestions: {data.get('development_suggestions')}")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_get_specific_type_zh():
    """测试获取特定MBTI类型描述（中文）"""
    print_section("Test 4: GET /api/mbti/descriptions/ENFP?language=zh")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions/ENFP?language=zh")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"   Type: {data.get('type')}")
            print(f"   Title: {data.get('title')}")
            print(f"   Description: {data.get('description')}")
            print(f"   Characteristics: {', '.join(data.get('characteristics', []))}")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_invalid_type():
    """测试无效的MBTI类型"""
    print_section("Test 5: GET /api/mbti/descriptions/INVALID (should fail)")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions/INVALID")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 400:
            print(f"✅ Correctly returned 400 for invalid type")
            print(f"   Error: {response.json().get('error')}")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_nonexistent_type():
    """测试不存在但格式有效的MBTI类型"""
    print_section("Test 6: GET /api/mbti/descriptions/EEEE (should fail)")

    try:
        response = requests.get(f"{BASE_URL}/api/mbti/descriptions/EEEE")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 404:
            print(f"✅ Correctly returned 404 for non-existent type")
            print(f"   Error: {response.json().get('error')}")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {e}")

def test_all_16_types():
    """测试所有16个MBTI类型"""
    print_section("Test 7: Verify all 16 MBTI types are available")

    all_types = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ',
                 'ISTP', 'ISFP', 'INFP', 'INTP',
                 'ESTP', 'ESFP', 'ENFP', 'ENTP',
                 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

    success_count = 0

    for mbti_type in all_types:
        try:
            response = requests.get(f"{BASE_URL}/api/mbti/descriptions/{mbti_type}?language=en")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {mbti_type:4s} - {data.get('title')}")
                success_count += 1
            else:
                print(f"❌ {mbti_type:4s} - Failed (status: {response.status_code})")
        except Exception as e:
            print(f"❌ {mbti_type:4s} - Exception: {e}")

    print(f"\nResult: {success_count}/16 types available")

    if success_count == 16:
        print("✅ All 16 MBTI types are accessible!")
    else:
        print(f"⚠️  Only {success_count} types are accessible")

def main():
    print("\n🚀 Starting MBTI Descriptions API Tests...")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Make sure the Flask server is running on port 5000")

    # Run all tests
    test_get_all_descriptions_en()
    test_get_all_descriptions_zh()
    test_get_specific_type_en()
    test_get_specific_type_zh()
    test_invalid_type()
    test_nonexistent_type()
    test_all_16_types()

    print("\n" + "="*70)
    print("  ✅ All tests completed!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
