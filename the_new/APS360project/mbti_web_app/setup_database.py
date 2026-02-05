"""
数据库设置和初始化脚本
"""

import os
from database import init_database, get_prediction_statistics

print("=" * 60)
print("PostgreSQL 数据库初始化")
print("=" * 60)

# 步骤1: 检查环境变量
database_url = os.getenv('DATABASE_URL')
if database_url:
    print(f"✅ 使用环境变量中的数据库URL")
else:
    print("⚠️  未设置 DATABASE_URL 环境变量")
    print("   使用默认: postgresql://postgres:password@localhost:5432/mbti_predictions")
    print()
    print("   设置方法:")
    print("   PowerShell: $env:DATABASE_URL='your_database_url'")
    print("   Linux/Mac: export DATABASE_URL='your_database_url'")
    print()

# 步骤2: 初始化数据库
print("\n正在创建数据库表...")
if init_database():
    print("✅ 数据库初始化成功！")
    print()
    print("创建的表:")
    print("  - prediction_data (预测数据)")
    print("  - question_usage (问题使用记录)")
    print("  - user_sessions (用户会话)")
else:
    print("❌ 数据库初始化失败")
    print()
    print("请检查:")
    print("  1. PostgreSQL 是否已安装并运行")
    print("  2. 数据库 'mbti_predictions' 是否存在")
    print("  3. 用户名和密码是否正确")
    print()
    print("创建数据库命令 (PostgreSQL):")
    print("  psql -U postgres")
    print("  CREATE DATABASE mbti_predictions;")
    exit(1)

# 步骤3: 测试连接
print("\n正在测试数据库连接...")
try:
    stats = get_prediction_statistics()
    if stats is not None:
        print("✅ 数据库连接成功！")
        print(f"\n当前数据库统计:")
        print(f"  总预测数: {stats['total_predictions']}")
        if stats['most_common']:
            print(f"  最常见类型: {stats['most_common']}")
    else:
        print("⚠️  数据库连接成功但查询失败")
except Exception as e:
    print(f"❌ 数据库连接失败: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ 数据库设置完成！")
print("=" * 60)
print("\n现在可以运行: python app.py")
