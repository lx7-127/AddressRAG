import re


def augment_with_full_precision(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    augmented_results = []

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. 清理 前缀并去除末尾句点
        clean_line = re.sub(r'^\\s*', '', line).rstrip('.')

        # 2. 鲁棒解析：按逗号分割字段
        segments = clean_line.split(',')
        pairs = {}
        for seg in segments:
            if '是' in seg:
                # 仅在第一个“是”处分割，确保Value中即使有特殊字符也能保留
                parts = seg.split('是', 1)
                pairs[parts[0].strip()] = parts[1].strip()

        # 3. 字段提取（匹配您的数据键名）
        name = pairs.get('地名', pairs.get('名称', '未知'))
        cat_l = pairs.get('地名大类', pairs.get('大类', '未知'))
        cat_m = pairs.get('地名中类', pairs.get('中类', '未知'))
        cat_s = pairs.get('地名小类', pairs.get('小类', '未知'))
        addr = pairs.get('地址', '未知')
        prov = pairs.get('省', '未知')
        city = pairs.get('市', '未知')
        dist = pairs.get('区', '未知')
        lng = pairs.get('WGS84_经度', '0.0')
        lat = pairs.get('WGS84_纬度', '0.0')

        # 4. 注入语义增强模板
        augmented = (
            f"地点名称：{name}\n"
            f"地理归属：该地点行政上隶属于{prov}{city}{dist}，其具体的详细物理地址位于{addr}。\n"
            f"空间定位：在 WGS84 坐标系下，该位置的地理经度标记为 {lng}，纬度标记为 {lat}。\n"
            f"分类特征：在分类体系中，{name}属于“{cat_l}”行业范畴，并进一步细分为“{cat_m}”以及“{cat_s}”属性。\n"
            f"空间逻辑：作为{dist}境内的一个重要节点，{name} 与周边地理实体共同构建了局部的空间服务拓扑，承载着特定的社会经济功能。"
        )
        augmented_results.append(augmented)

    # 5. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n---\n\n".join(augmented_results))

# 使用方法：
augment_with_full_precision('data/1000.txt', 'augmented_full_precision0509.txt')