"""
数组类型注解模块 - 为JAX和PyTorch提供统一的类型检查

本模块提供了类型注解和运行时类型检查工具，基于jaxtyping库。

主要功能:
1. 统一的数组类型：同时支持JAX数组和PyTorch张量
2. 形状和dtype类型检查：使用装饰器进行运行时验证
3. PyTree类型支持：处理嵌套的数据结构
4. 类型检查控制：可以临时禁用类型检查

核心导出:
- Array: JAX数组或PyTorch张量的联合类型
- 类型注解：Float, Int, Bool, Real, UInt8等
- typecheck装饰器：为函数添加运行时类型检查
- check_pytree_equality: 验证PyTree结构和形状
- disable_typechecking: 临时禁用类型检查的上下文管理器

使用示例:
    @typecheck
    def process_image(
        image: Float[Array, "batch height width channels"]
    ) -> Float[Array, "batch features"]:
        '''处理图像，返回特征向量'''
        ...

    # 验证PyTree结构
    check_pytree_equality(
        expected=model_spec,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True
    )

关键补丁:
- 修复了jaxtyping在处理自定义PyTree节点时的问题
- 跳过JAX内部tree_util调用时的类型检查，避免初始化错误

技术细节:
- 使用beartype进行运行时类型检查
- 支持维度名称和广播语义
- 提供友好的类型不匹配错误信息
"""
import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

# 修复jaxtyping以处理 https://github.com/patrick-kidger/jaxtyping/issues/277
# 问题描述：自定义PyTree节点在初始化时可能使用任意类型（如 jax.ShapeDtypeStruct、
# jax.Sharding，甚至 <object>），这是由于JAX的追踪操作导致的。
# 解决方案：当栈跟踪包含 jax._src.tree_util 时跳过类型检查，
# 这种情况只会在树展开（tree unflattening）期间发生。
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001

# 重新定义Array类型，同时包含JAX数组和PyTorch张量
Array = jax.Array | torch.Tensor


def _check_dataclass_annotations(self, typechecker):
    """修补的dataclass注解检查函数

    在JAX内部tree_util或Flax编译期间跳过类型检查，
    避免因临时使用特殊对象类型而导致的错误。

    参数:
        self: jaxtyping装饰器实例
        typechecker: 类型检查函数

    返回:
        类型检查结果，或None（如果跳过）
    """
    if not any(
        frame.frame.f_globals.get("__name__") in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


# 应用补丁
jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

# 类型别名定义
KeyArrayLike: TypeAlias = jax.typing.ArrayLike
"""JAX随机数生成器密钥的类型别名"""

Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]
"""模型参数的类型别名，表示浮点数PyTree"""

T = TypeVar("T")
"""通用类型变量"""


# 运行时类型检查装饰器
def typecheck(t: T) -> T:
    """运行时类型检查装饰器

    使用jaxtyping和beartype为函数添加类型检查。
    检查函数参数和返回值是否符合类型注解。

    参数:
        t: 要装饰的函数或类

    返回:
        添加了类型检查的函数或类

    示例:
        @typecheck
        def add_arrays(
            a: Float[Array, "n"],
            b: Float[Array, "n"]
        ) -> Float[Array, "n"]:
            return a + b

        # 正确调用
        result = add_arrays(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

        # 错误调用会抛出类型错误
        # result = add_arrays(jnp.array([1.0]), jnp.array([2.0, 3.0]))  # 形状不匹配

    注意:
        - 类型检查在运行时执行，会有一定性能开销
        - 可以使用disable_typechecking临时禁用检查
        - 支持形状、dtype和结构检查
    """
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    """临时禁用类型检查的上下文管理器

    在性能关键的代码段中可以禁用类型检查以提高速度。

    使用示例:
        @typecheck
        def expensive_function(x: Float[Array, "n"]) -> Float[Array, "n"]:
            return x * 2

        # 正常情况下会进行类型检查
        result1 = expensive_function(jnp.array([1.0, 2.0]))

        # 临时禁用类型检查
        with disable_typechecking():
            # 这里不会进行类型检查，速度更快
            result2 = expensive_function(jnp.array([3.0, 4.0]))

    注意:
        - 禁用后会跳过所有类型检查，可能导致难以调试的错误
        - 仅在确认类型正确且需要优化性能时使用
        - 上下文结束后会恢复之前的检查状态
    """
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """检查两个PyTree是否具有相同的结构，并可选检查形状和dtype

    相比直接使用 jax.tree.map，此函数在结构不匹配时提供更友好的错误信息。

    参数:
        expected: 期望的PyTree结构
        got: 实际的PyTree结构
        check_shapes: 是否检查数组形状是否匹配（默认False）
        check_dtypes: 是否检查数组dtype是否匹配（默认False）

    抛出:
        ValueError: 当PyTree结构、形状或dtype不匹配时

    示例:
        # 检查模型参数结构
        expected_params = {"weight": jnp.zeros((10, 5)), "bias": jnp.zeros(5)}
        loaded_params = {"weight": jnp.ones((10, 5)), "bias": jnp.ones(5)}

        # 只检查结构
        check_pytree_equality(expected=expected_params, got=loaded_params)

        # 同时检查形状
        check_pytree_equality(
            expected=expected_params,
            got=loaded_params,
            check_shapes=True
        )

        # 错误示例：形状不匹配
        wrong_params = {"weight": jnp.ones((5, 10)), "bias": jnp.ones(5)}
        # check_pytree_equality(expected=expected_params, got=wrong_params, check_shapes=True)
        # ValueError: Shape mismatch at weight: expected (10, 5), got (5, 10)

    注意:
        - 默认只检查树结构（键和嵌套层次）
        - 启用check_shapes时会逐个比较叶子节点的形状
        - 启用check_dtypes时会逐个比较叶子节点的数据类型
        - 错误信息包含完整的keypath，便于定位问题
    """

    # 首先检查树结构是否一致
    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    # 如果需要，检查形状和/或dtype
    if check_shapes or check_dtypes:

        def check(kp, x, y):
            """逐叶检查形状和dtype

            参数:
                kp: keypath（叶子节点的路径）
                x: 期望的叶子节点
                y: 实际的叶子节点
            """
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)
