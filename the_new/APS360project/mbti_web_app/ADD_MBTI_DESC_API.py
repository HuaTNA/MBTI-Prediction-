"""
这个文件包含需要添加到app.py的代码片段
将这些代码添加到app.py的相应位置
"""

# =============================================================================
# 1. 添加到 load_question_banks() 函数后面
# =============================================================================

def load_mbti_descriptions():
    """加载MBTI类型描述"""
    global mbti_descriptions_en, mbti_descriptions_zh

    try:
        # 英文描述
        en_file = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_en.json')
        if os.path.exists(en_file):
            with open(en_file, 'r', encoding='utf-8') as f:
                mbti_descriptions_en = json.load(f)
            logger.info(f"✅ 成功加载英文MBTI描述: {len(mbti_descriptions_en.get('types', {}))} 个类型")
        else:
            logger.warning(f"⚠️  英文MBTI描述文件不存在: {en_file}")
            mbti_descriptions_en = {"types": {}}

        # 中文描述
        zh_file = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_zh.json')
        if os.path.exists(zh_file):
            with open(zh_file, 'r', encoding='utf-8') as f:
                mbti_descriptions_zh = json.load(f)
            logger.info(f"✅ 成功加载中文MBTI描述: {len(mbti_descriptions_zh.get('types', {}))} 个类型")
        else:
            logger.warning(f"⚠️  中文MBTI描述文件不存在: {zh_file}")
            mbti_descriptions_zh = {"types": {}}

        return True

    except Exception as e:
        logger.error(f"❌ 加载MBTI描述时出错: {e}")
        mbti_descriptions_en = {"types": {}}
        mbti_descriptions_zh = {"types": {}}
        return False


# =============================================================================
# 2. 在 initialize_app() 函数中添加（在 load_question_banks() 之后）
# =============================================================================

    # 加载MBTI描述
    descriptions_loaded = load_mbti_descriptions()

    # 在状态输出中添加
    logger.info(f"  {'✅' if descriptions_loaded else '❌'} MBTI描述: {'已加载' if descriptions_loaded else '未加载'}")


# =============================================================================
# 3. 添加API端点（在 /api/questions/stats 后面）
# =============================================================================

@app.route('/api/mbti/descriptions', methods=['GET'])
def get_mbti_descriptions():
    """
    获取所有MBTI类型描述

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
        JSON: {
            "version": "1.0",
            "language": "en",
            "types": {
                "ISTJ": {...},
                "ISFJ": {...},
                ...
            }
        }
    """
    try:
        language = request.args.get('language', 'en')

        # 选择对应语言的描述
        descriptions = mbti_descriptions_zh if language == 'zh' else mbti_descriptions_en

        if descriptions is None or not descriptions.get('types'):
            logger.error(f"MBTI描述未初始化 (language={language})")
            return jsonify({
                'error': 'MBTI descriptions not initialized',
                'language': language
            }), 500

        logger.info(f"获取MBTI描述 (语言={language}, 类型数={len(descriptions.get('types', {}))})")

        return jsonify(descriptions)

    except Exception as e:
        logger.error(f"获取MBTI描述时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mbti/descriptions/<mbti_type>', methods=['GET'])
def get_mbti_description_by_type(mbti_type):
    """
    根据MBTI类型获取描述

    路径参数:
        mbti_type (str): MBTI类型，如 'INTJ', 'ENFP' 等

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
        JSON: {
            "type": "INTJ",
            "title": "The Mastermind",
            "description": "...",
            "characteristics": [...],
            "career_paths": "...",
            "development_suggestions": "..."
        }
    """
    try:
        language = request.args.get('language', 'en')
        mbti_type = mbti_type.upper()

        # 验证MBTI类型格式
        if len(mbti_type) != 4 or not all(c in 'IESTFNJP' for c in mbti_type):
            return jsonify({'error': 'Invalid MBTI type format'}), 400

        # 选择对应语言的描述
        descriptions = mbti_descriptions_zh if language == 'zh' else mbti_descriptions_en

        if descriptions is None:
            return jsonify({'error': 'MBTI descriptions not initialized'}), 500

        # 获取特定类型的描述
        type_desc = descriptions.get('types', {}).get(mbti_type)

        if type_desc:
            logger.info(f"获取MBTI描述: {mbti_type} (语言={language})")
            result = {
                'type': mbti_type,
                **type_desc
            }
            return jsonify(result)
        else:
            return jsonify({'error': f'MBTI type {mbti_type} not found'}), 404

    except Exception as e:
        logger.error(f"获取MBTI描述 {mbti_type} 时出错: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 使用说明
# =============================================================================

"""
将上面的代码添加到 app.py 的相应位置后，API就可以使用了：

测试：
1. 获取所有MBTI描述（英文）：
   curl http://localhost:5000/api/mbti/descriptions?language=en

2. 获取所有MBTI描述（中文）：
   curl http://localhost:5000/api/mbti/descriptions?language=zh

3. 获取特定类型（INTJ）的描述：
   curl http://localhost:5000/api/mbti/descriptions/INTJ?language=en

4. 获取特定类型（ENFP）的中文描述：
   curl http://localhost:5000/api/mbti/descriptions/ENFP?language=zh
"""
