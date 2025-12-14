"""
归一化统计模块 - 数据归一化的统计信息计算和管理

本模块提供了数据归一化所需的统计信息计算和存储功能。

核心类:
1. NormStats: 归一化统计数据类（均值、标准差、分位数）
2. RunningStats: 运行时统计计算器，支持增量更新

主要功能:
- 计算和维护归一化所需的统计量
- 支持两种归一化方式：
  * Z-score归一化：使用均值和标准差
  * 分位数归一化：使用1%和99%分位数
- 增量统计更新：无需存储所有数据
- 序列化和反序列化：方便保存和加载统计信息

使用场景:
- 训练时：计算数据集的归一化统计
- 推理时：加载预先计算的统计信息进行归一化

示例:
    # 训练时收集统计
    stats = RunningStats()
    for batch in dataset:
        stats.update(batch["actions"])
    norm_stats = stats.get_statistics()

    # 保存统计信息
    save("./checkpoints", {"actions": norm_stats})

    # 推理时加载统计信息
    loaded_stats = load("./checkpoints")
"""
import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    """归一化统计数据类

    存储用于数据归一化和反归一化的统计信息。

    属性:
        mean: 均值，用于Z-score归一化
        std: 标准差，用于Z-score归一化
        q01: 1%分位数，用于分位数归一化（可选）
        q99: 99%分位数，用于分位数归一化（可选）

    归一化方法:
        Z-score归一化: normalized = (x - mean) / std
        分位数归一化: normalized = (x - q01) / (q99 - q01) * 2 - 1
    """

    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1% quantile
    q99: numpydantic.NDArray | None = None  # 99% quantile


class RunningStats:
    """运行时统计计算器 - 增量计算批次向量的统计信息

    此类支持在线（增量）方式计算统计量，无需存储所有历史数据。
    特别适用于大规模数据集的统计信息收集。

    支持的统计量:
    - 均值 (mean)
    - 标准差 (std)
    - 最小值/最大值 (min/max)
    - 分位数 (quantiles)

    实现方法:
    - 均值和方差：使用Welford's算法进行增量更新
    - 分位数：使用动态直方图近似计算

    注意事项:
    - 所有输入向量必须具有相同的最后一个维度
    - 分位数通过直方图近似，精度取决于bin数量
    """

    def __init__(self):
        """初始化运行时统计计算器

        内部状态:
        - _count: 已处理的样本总数
        - _mean: 当前均值估计
        - _mean_of_squares: 平方的均值（用于计算方差）
        - _min/_max: 最小值和最大值
        - _histograms: 每个维度的直方图（用于分位数计算）
        - _bin_edges: 直方图的bin边界
        """
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # 用于实时计算分位数的bin数量

    def update(self, batch: np.ndarray) -> None:
        """使用一批向量更新运行时统计

        使用增量算法更新所有统计量，避免存储历史数据。

        参数:
            batch: 输入数组，除最后一个维度外的所有维度都视为批次维度
                  形状: [..., feature_dim]
                  例如: [batch, time, feature] 或 [batch, feature]

        注意:
            - 向量长度（最后一个维度）必须与首次更新时一致
            - 如果最小值/最大值发生变化，会自动调整直方图

        抛出:
            ValueError: 如果向量长度与之前不匹配
        """
        # 将所有批次维度展平为单一批次维度
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._count == 0:
            # 首次更新：初始化所有统计量
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            # 为每个特征维度创建直方图
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            # 设置bin边界，添加小偏移避免边界值问题
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            # 后续更新：检查维度一致性
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            # 更新最小值和最大值
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            # 如果范围改变，需要调整直方图
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        # 计算当前批次的统计量
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # 使用增量公式更新全局均值和平方均值
        # 新均值 = 旧均值 + (批次均值 - 旧均值) * (批次大小 / 总数)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        # 更新直方图
        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """计算并返回当前的归一化统计信息

        返回:
            NormStats对象，包含:
            - mean: 均值
            - std: 标准差
            - q01: 1%分位数
            - q99: 99%分位数

        抛出:
            ValueError: 如果样本数量少于2（无法计算方差）

        注意:
            - 标准差使用无偏估计（Bessel's correction已包含在方差计算中）
            - 分位数通过直方图近似计算，可能与精确值略有差异
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        # 计算方差: Var(X) = E[X²] - E[X]²
        variance = self._mean_of_squares - self._mean**2
        # 确保方差非负（避免数值误差导致负值）
        stddev = np.sqrt(np.maximum(0, variance))

        # 计算1%和99%分位数
        q01, q99 = self._compute_quantiles([0.01, 0.99])

        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """当最小值或最大值改变时调整直方图

        当新数据扩展了数据范围时，需要重新分配直方图的bin。
        此方法使用加权重分配来保持计数的近似准确性。
        """
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # 将旧直方图的计数重新分配到新的bin中
            # 使用旧bin的中心点和权重进行重新分配
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """使用新向量更新直方图

        参数:
            batch: 形状为 [num_samples, feature_dim] 的批次数据
        """
        # 为每个特征维度更新直方图
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """基于直方图计算分位数

        使用累积直方图进行分位数的近似计算。

        参数:
            quantiles: 要计算的分位数列表，例如 [0.01, 0.5, 0.99]

        返回:
            对应分位数的值列表

        算法:
        1. 计算每个特征维度的累积直方图
        2. 找到累积计数超过目标计数的第一个bin
        3. 返回该bin的左边界作为分位数估计
        """
        results = []
        for q in quantiles:
            # 计算目标累积计数
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                # 计算累积分布
                cumsum = np.cumsum(hist)
                # 找到第一个超过目标计数的bin索引
                idx = np.searchsorted(cumsum, target_count)
                # 使用该bin的左边界作为分位数估计
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    """归一化统计字典的内部包装类

    用于Pydantic的序列化和反序列化。
    """

    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """将归一化统计序列化为JSON字符串

    参数:
        norm_stats: 归一化统计字典，键为字段名，值为NormStats对象

    返回:
        格式化的JSON字符串（带缩进）
    """
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """从JSON字符串反序列化归一化统计

    参数:
        data: JSON格式的字符串

    返回:
        归一化统计字典

    抛出:
        json.JSONDecodeError: 如果JSON格式无效
        pydantic.ValidationError: 如果数据结构不匹配
    """
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """将归一化统计保存到目录

    统计信息会保存为 directory/norm_stats.json

    参数:
        directory: 目标目录路径
        norm_stats: 要保存的归一化统计字典

    注意:
        如果目录不存在会自动创建
    """
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """从目录加载归一化统计

    从 directory/norm_stats.json 加载统计信息

    参数:
        directory: 包含统计文件的目录路径

    返回:
        归一化统计字典

    抛出:
        FileNotFoundError: 如果统计文件不存在
    """
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
