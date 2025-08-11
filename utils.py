def list_stats(data):
    """
        计算均值及半范围（误差范围）
        输入示例：[0.5, 0.6] → 均值0.55 ± 0.05

        参数：
        data -- 数值型列表（至少包含一个元素）

        返回：
        (均值, 半范围)
        """
    # 数据校验
    if not data:
        raise ValueError("输入列表不能为空")
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("列表必须包含数值类型")

    # 计算核心指标
    mean = sum(data) / len(data)
    data_range = max(data) - min(data)
    half_range = data_range / 2

    return mean, half_range

