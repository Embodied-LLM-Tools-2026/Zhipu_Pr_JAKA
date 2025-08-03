# ================================
# 拼音匹配模块
# ================================

class PinyinMatcher:
    """拼音匹配器，支持前后鼻音和平翘舌音模糊匹配"""
    
    def __init__(self):
        # 前后鼻音映射表
        self.nasal_map = {
            'an': 'ang', 'en': 'eng', 'in': 'ing', 'un': 'ong',
            'ang': 'an', 'eng': 'en', 'ing': 'in', 'ong': 'un'
        }
        
        # 平舌音和翘舌音映射表
        self.tongue_map = {
            'z': 'zh', 'c': 'ch', 's': 'sh',
            'zh': 'z', 'ch': 'c', 'sh': 's'
        }
        
        # 支持多个唤醒词
        self.wake_words = ["小拓", "小兔"]
        self.wake_pinyin_variants = {}
        
        for wake_word in self.wake_words:
            self.wake_pinyin_variants[wake_word] = self._get_pinyin_variants(wake_word)
        
        print(f"唤醒词已初始化: {', '.join(self.wake_words)}")
    
    def _get_pinyin_variants(self, text: str) -> list:
        """获取文本的拼音变体"""
        # if not deps.pypinyin_available:
        #     return []
        
        from pypinyin import pinyin, Style
        
        original_pinyin = pinyin(text, style=Style.NORMAL, heteronym=False)
        
        variants = []
        for char_pinyin_list in original_pinyin:
            char_pinyin = char_pinyin_list[0].lower()
            char_variants = [char_pinyin]
            
            # 添加前后鼻音变体
            for original, variant in self.nasal_map.items():
                if char_pinyin.endswith(original):
                    new_variant = char_pinyin[:-len(original)] + variant
                    if new_variant not in char_variants:
                        char_variants.append(new_variant)
            
            # 添加平翘舌音变体
            for original, variant in self.tongue_map.items():
                if char_pinyin.startswith(original):
                    new_variant = variant + char_pinyin[len(original):]
                    if new_variant not in char_variants:
                        char_variants.append(new_variant)
            
            # 组合变体
            combined_variants = []
            for base_variant in char_variants[:]:
                for original, variant in self.nasal_map.items():
                    if base_variant.endswith(original):
                        combined_variant = base_variant[:-len(original)] + variant
                        if combined_variant not in char_variants:
                            combined_variants.append(combined_variant)
                
                for original, variant in self.tongue_map.items():
                    if base_variant.startswith(original):
                        combined_variant = variant + base_variant[len(original):]
                        if combined_variant not in char_variants:
                            combined_variants.append(combined_variant)
            
            char_variants.extend(combined_variants)
            variants.append(char_variants)
        
        return variants
    
    def _generate_pinyin_combinations(self, variants: list) -> list:
        """生成所有可能的拼音组合"""
        if not variants:
            return []
        
        def combine(index: int, current: list) -> list:
            if index == len(variants):
                return [''.join(current)]
            
            results = []
            for variant in variants[index]:
                results.extend(combine(index + 1, current + [variant]))
            return results
        
        return combine(0, [])
    
    def detect_wake_word(self, text: str) -> tuple:
        """检测文本中是否包含唤醒词，返回(是否检测到, 剩余文本)"""
        if not text:
            return False, text
        
        # 清理文本
        import re
        cleaned_text = re.sub(r'[^\u4e00-\u9fff，。！？、]', '', text)
        
        if not cleaned_text:
            return False, text
        
        # 获取输入文本的拼音
        from pypinyin import pinyin, Style
        input_pinyin = pinyin(cleaned_text, style=Style.NORMAL, heteronym=False)
        input_pinyin_str = ''.join([p[0].lower() for p in input_pinyin])
        
        # 检查每个唤醒词
        for wake_word in self.wake_words:
            # 生成目标拼音的所有组合
            wake_combinations = self._generate_pinyin_combinations(self.wake_pinyin_variants[wake_word])
            
            # 检查是否匹配
            for wake_combo in wake_combinations:
                if wake_combo in input_pinyin_str:
                    print(f"🎯 检测到唤醒词: '{wake_word}'")
                    
                    # 找到唤醒词在原文本中的位置
                    wake_word_index = cleaned_text.find(wake_word)
                    if wake_word_index != -1:
                        remaining_text = cleaned_text[wake_word_index + len(wake_word):].strip('，。！？、')
                        print(f"剩余指令文本: '{remaining_text}'")
                        return True, remaining_text
                    else:
                        return True, cleaned_text
        
        return False, text

    def detect_dismiss_command(self, text: str) -> bool:
        """检测退下指令"""
        if not text:
            return False
        
        dismiss_keywords = [
            "退下", "休息", "睡觉", "回去休息", "去休息", 
            "暂停", "停止", "结束", "再见", "拜拜",
            "你可以休息了", "你可以去休息了", "没事了", "辛苦了","退一下"
        ]
        
        text_cleaned = text.strip().replace(" ", "")
        for keyword in dismiss_keywords:
            if keyword in text_cleaned:
                print(f"🛌 检测到退下指令: '{keyword}'")
                return True
        
        return False
