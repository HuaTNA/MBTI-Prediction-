# 📝 How to Add Questions to the Question Bank

## 问题库文件位置

```
e:\APS360_project\MBTI-Prediction-\the_new\APS360project\mbti_web_app\data\
├── questions_en.json  ← 英文问题库 (当前10个问题)
└── questions_zh.json  ← 中文问题库 (当前10个问题)
```

---

## 📋 添加新问题的步骤

### 方法1: 直接编辑JSON文件（推荐）

#### 步骤1: 打开文件

在VSCode或任何文本编辑器中打开：
- 英文: `data/questions_en.json`
- 中文: `data/questions_zh.json`

#### 步骤2: 添加新问题

在 `"questions"` 数组的末尾添加新问题：

```json
{
  "version": "1.0",
  "language": "en",
  "last_updated": "2026-02-05",
  "description": "MBTI personality test questions - English version",
  "questions": [
    {
      "id": 1,
      "question": "About Social Interactions",
      "description": "...",
      "dimension": "I/E",
      "version": "1.0",
      "language": "en",
      "active": true
    },
    // ... 其他问题 ...
    {
      "id": 10,
      "question": "About Communication Style",
      "description": "...",
      "dimension": "T/F",
      "version": "1.0",
      "language": "en",
      "active": true
    },
    // ========== 在这里添加新问题 ==========
    {
      "id": 11,                                    // 新的ID（递增）
      "question": "关于工作环境偏好",                 // 问题标题
      "description": "你更喜欢什么样的工作环境？是结构化、有明确规则的环境，还是灵活、自由度高的环境？", // 详细描述
      "dimension": "J/P",                          // MBTI维度
      "version": "1.0",                            // 版本号
      "language": "en",                            // 语言（en或zh）
      "active": true                               // 是否启用
    }
    // 注意：最后一个问题后面不要加逗号！
  ]
}
```

#### 步骤3: 验证JSON格式

确保：
- ✅ 每个问题之间用**逗号**分隔
- ✅ **最后一个问题**后面**没有逗号**
- ✅ 所有引号都是**双引号** `"`
- ✅ `id` 是唯一的（不重复）

可以使用在线JSON验证器：https://jsonlint.com/

#### 步骤4: 重启服务器

```bash
# Ctrl+C 停止服务器
python app.py
```

服务器会自动重新加载问题库：
```
✅ 成功加载英文问题库: 11 个问题  ← 看到数量增加了！
✅ 成功加载中文问题库: 10 个问题
```

---

## 📚 问题字段说明

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| **id** | 整数 | ✅ 是 | 问题唯一标识符（递增） | `11` |
| **question** | 字符串 | ✅ 是 | 问题标题（简短） | `"About Work Environment"` |
| **description** | 字符串 | ✅ 是 | 问题详细描述（用户看到的完整问题） | `"What type of work environment do you prefer..."` |
| **dimension** | 字符串 | ✅ 是 | MBTI维度（必须是以下之一）<br>• `I/E` - 内向/外向<br>• `S/N` - 感觉/直觉<br>• `T/F` - 思考/情感<br>• `J/P` - 判断/感知 | `"J/P"` |
| **version** | 字符串 | 否 | 问题版本（用于A/B测试） | `"1.0"` |
| **language** | 字符串 | ✅ 是 | 语言代码 | `"en"` 或 `"zh"` |
| **active** | 布尔值 | 否 | 是否启用（false会被跳过） | `true` |

---

## 🎯 MBTI维度说明

确保问题平衡覆盖所有4个维度：

| 维度 | 测量内容 | 问题重点 |
|------|---------|---------|
| **I/E** | 内向 vs 外向 | 能量来源、社交偏好 |
| **S/N** | 感觉 vs 直觉 | 信息获取方式、思维模式 |
| **T/F** | 思考 vs 情感 | 决策方式、价值观 |
| **J/P** | 判断 vs 感知 | 生活方式、计划性 |

**推荐分布**：
- 每个维度至少 **2-3个问题**
- 总数 **10-30个问题**（太多会让用户疲劳）
- 当前系统随机选择 **3个问题**（每个维度尽量平衡）

---

## 📝 问题编写最佳实践

### ✅ 好的问题

```json
{
  "id": 11,
  "question": "About Conflict Resolution",
  "description": "When there's a disagreement in your team, what's your natural approach? Do you focus on finding the logical solution that works best objectively, or do you prioritize maintaining harmony and considering everyone's feelings?",
  "dimension": "T/F",
  "version": "1.0",
  "language": "en",
  "active": true
}
```

**为什么好**：
- ✅ 清晰的对比选项（逻辑 vs 和谐）
- ✅ 具体的场景（团队分歧）
- ✅ 开放式表述（让用户详细回答）
- ✅ 明确对应MBTI维度（T/F）

### ❌ 不好的问题

```json
{
  "id": 12,
  "question": "Do you like planning?",
  "description": "Do you like to plan things?",
  "dimension": "J/P",
  "version": "1.0",
  "language": "en",
  "active": true
}
```

**为什么不好**：
- ❌ 太简单（是/否问题）
- ❌ 描述和标题重复
- ❌ 没有提供具体场景
- ❌ 难以引导出有意义的回答

---

## 🌍 添加中文问题

编辑 `questions_zh.json`，格式相同：

```json
{
  "id": 11,
  "question": "关于冲突解决",
  "description": "当团队出现分歧时，你的自然反应是什么？你会专注于找到客观上最有效的逻辑解决方案，还是优先考虑维护和谐并照顾每个人的感受？",
  "dimension": "T/F",
  "version": "1.0",
  "language": "zh",
  "active": true
}
```

**注意**：
- 中英文问题的 `id` 可以相同（表示同一个问题的不同语言版本）
- 或者使用不同的 `id`（独立的问题集）

---

## 🧪 测试新问题

### 方法1: API测试

```bash
# 查看所有英文问题
curl http://localhost:5000/api/questions/stats?language=en

# 获取3个随机问题
curl http://localhost:5000/api/questions?language=en&count=3

# 查看特定问题
curl http://localhost:5000/api/questions/11?language=en
```

### 方法2: 浏览器测试

1. 打开：http://localhost:5000
2. 按F12打开控制台
3. 点击"Start MBTI Test"
4. 查看控制台输出：
   ```javascript
   ✅ Loaded 3 questions (v1.0, session: ...)
   Questions by dimension: ['I/E', 'T/F', 'J/P']  // 看看是否包含新问题
   ```

### 方法3: 检查日志

服务器启动时会显示：
```
✅ 成功加载英文问题库: 11 个问题  ← 数量正确
✅ 成功加载中文问题库: 10 个问题
```

---

## 🔄 版本管理（高级）

### 创建问题的新版本（A/B测试）

```json
// 原版本
{
  "id": 5,
  "question": "About Stress Management",
  "description": "When facing stress, do you stay calm and analyze...",
  "dimension": "T/F",
  "version": "1.0",
  "active": true
}

// 新版本（测试不同措辞）
{
  "id": 5,
  "question": "About Stress Management",
  "description": "In stressful situations, do you prefer to step back and think logically...",
  "dimension": "T/F",
  "version": "1.1",
  "active": false  // 先禁用，测试后再启用
}
```

### 禁用表现不佳的问题

将 `active` 设为 `false`：

```json
{
  "id": 7,
  "question": "Some poorly performing question",
  "description": "...",
  "dimension": "S/N",
  "version": "1.0",
  "active": false  // ← 系统会跳过这个问题
}
```

---

## 📊 问题数量建议

| 用途 | 推荐数量 | 说明 |
|------|---------|------|
| **最小集合** | 4个 | 每个维度1个（紧急备用） |
| **快速测试** | 10-12个 | 每个维度2-3个 |
| **标准测试** | 20-30个 | 每个维度5-8个（推荐） |
| **完整测试** | 40-60个 | 每个维度10-15个 |

**当前配置**：系统随机选择 **3个问题**（可在 `index.html` 中修改）

---

## 🚨 常见错误

### 错误1: JSON格式错误

```json
{
  "id": 11,
  "question": "Test",
  "description": "...",
  "dimension": "T/F",
  "version": "1.0",
  "language": "en",
  "active": true
},  // ← 错误！最后一个问题不能有逗号
```

**修复**: 删除最后的逗号

### 错误2: ID重复

```json
{"id": 5, ...},
{"id": 5, ...}  // ← 错误！ID重复
```

**修复**: 使用唯一的ID

### 错误3: 维度拼写错误

```json
{"dimension": "IE"}  // ← 错误！应该是 "I/E"
```

**修复**: 必须是 `I/E`, `S/N`, `T/F`, `J/P` 之一

### 错误4: 缺少必填字段

```json
{
  "id": 11,
  "question": "Test"
  // ← 错误！缺少 description, dimension, language
}
```

**修复**: 添加所有必填字段

---

## 💡 快速添加模板

复制此模板添加新问题：

```json
{
  "id": __NEXT_ID__,
  "question": "__问题标题__",
  "description": "__详细描述（用户会看到这个）__",
  "dimension": "__选择: I/E, S/N, T/F, J/P__",
  "version": "1.0",
  "language": "__en 或 zh__",
  "active": true
}
```

---

## 🎉 完成！

添加问题后：
1. ✅ 保存JSON文件
2. ✅ 验证JSON格式（jsonlint.com）
3. ✅ 重启服务器
4. ✅ 检查日志确认问题数量
5. ✅ 在浏览器测试

**现在您的问题库已更新，用户可以立即使用新问题！** 🚀
