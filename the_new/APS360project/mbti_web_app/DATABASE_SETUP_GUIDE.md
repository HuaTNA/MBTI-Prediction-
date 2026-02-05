# PostgreSQL 数据库设置指南

## 本地开发环境设置

### 1. 安装 PostgreSQL

#### Windows:
1. 下载: https://www.postgresql.org/download/windows/
2. 运行安装程序
3. 记住你设置的密码（默认用户是 `postgres`）
4. 默认端口: 5432

#### Mac (使用 Homebrew):
```bash
brew install postgresql
brew services start postgresql
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 2. 创建数据库

打开 PostgreSQL 命令行工具：

```bash
# Windows
psql -U postgres

# Mac/Linux
sudo -u postgres psql
```

在 psql 中执行：

```sql
CREATE DATABASE mbti_predictions;
\q
```

### 3. 配置数据库连接

设置环境变量（根据你的实际配置修改）：

#### PowerShell (Windows):
```powershell
$env:DATABASE_URL="postgresql://postgres:your_password@localhost:5432/mbti_predictions"
```

#### Bash (Linux/Mac):
```bash
export DATABASE_URL="postgresql://postgres:your_password@localhost:5432/mbti_predictions"
```

### 4. 安装 Python 依赖

```bash
pip install psycopg2-binary sqlalchemy
```

### 5. 初始化数据库表

```bash
python setup_database.py
```

应该看到：
```
✅ 数据库初始化成功！
创建的表:
  - prediction_data (预测数据)
  - question_usage (问题使用记录)
  - user_sessions (用户会话)
```

### 6. 运行应用

```bash
python app.py
```

---

## 云端部署设置

### 选项1: Heroku (推荐，简单)

1. **创建 Heroku 应用**
   ```bash
   heroku create your-app-name
   ```

2. **添加 PostgreSQL 附加组件**
   ```bash
   heroku addons:create heroku-postgresql:mini
   ```

3. **Heroku 会自动设置 DATABASE_URL 环境变量**
   - 无需手动配置！

4. **部署**
   ```bash
   git push heroku main
   ```

5. **初始化数据库**
   ```bash
   heroku run python setup_database.py
   ```

### 选项2: AWS RDS

1. **创建 RDS PostgreSQL 实例**
   - 登录 AWS Console
   - 进入 RDS
   - 创建 PostgreSQL 数据库

2. **获取连接信息**
   - 端点 (Endpoint)
   - 端口 (Port)
   - 数据库名称
   - 用户名/密码

3. **设置环境变量**
   ```bash
   export DATABASE_URL="postgresql://username:password@your-rds-endpoint.amazonaws.com:5432/mbti_predictions"
   ```

4. **配置安全组**
   - 允许你的应用服务器IP访问RDS

### 选项3: DigitalOcean Managed Database

1. **创建 Managed PostgreSQL 数据库**
   - 访问 DigitalOcean
   - 选择 Databases
   - 创建 PostgreSQL 集群

2. **获取连接字符串**
   - DigitalOcean 提供完整的连接URL

3. **设置环境变量**
   ```bash
   export DATABASE_URL="your_digitalocean_connection_string"
   ```

### 选项4: Google Cloud SQL

1. **创建 Cloud SQL PostgreSQL 实例**
2. **配置连接**
3. **设置环境变量**

---

## 环境变量格式

PostgreSQL 连接 URL 格式：

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

示例：
```
postgresql://postgres:mypassword@localhost:5432/mbti_predictions
postgresql://user:pass@my-db.amazonaws.com:5432/mbti_db
```

---

## 数据导出

导出所有数据用于训练：

```bash
python -c "from database import export_training_data; export_training_data()"
```

这会生成 `training_data.json` 文件，包含所有预测数据。

---

## 查看统计信息

```python
from database import get_prediction_statistics, get_recent_predictions

# 获取统计
stats = get_prediction_statistics()
print(f"总预测数: {stats['total_predictions']}")
print(f"MBTI分布: {stats['mbti_distribution']}")

# 查看最近10条预测
recent = get_recent_predictions(limit=10)
for pred in recent:
    print(f"{pred['timestamp']}: {pred['predicted_mbti']}")
```

---

## 常见问题

### Q: 连接被拒绝 (Connection refused)
A: 检查 PostgreSQL 是否正在运行：
```bash
# Windows
Get-Service postgresql*

# Linux/Mac
sudo systemctl status postgresql
```

### Q: 认证失败 (Authentication failed)
A: 检查用户名和密码是否正确

### Q: 数据库不存在
A: 运行 `CREATE DATABASE mbti_predictions;`

### Q: 云端部署时环境变量怎么设置？
A: 每个平台不同：
- **Heroku**: 自动设置，或 `heroku config:set DATABASE_URL=...`
- **AWS**: 在 Elastic Beanstalk 或 Lambda 配置中设置
- **DigitalOcean**: 在 App Platform 环境变量中设置

---

## 下一步

集成到应用后，你可以：
1. ✅ 自动保存每次预测
2. ✅ 收集用户数据
3. ✅ 导出数据训练新模型
4. ✅ 查看统计信息
5. ✅ 分析用户行为

运行 `python app.py` 开始使用！
