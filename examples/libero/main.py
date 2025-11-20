# 本脚本用于在 LIBERO 机器人操作任务套件上评估通过 WebSocket 推理服务提供的策略（policy）。
# 它会：
#     根据给定的任务集名称加载对应的 LIBERO 基准任务套件；
#     为每个任务读取固定的初始状态，构建无屏渲染环境（OffScreenRenderEnv）；
#     从环境中获取相机图像（主体视角与手部视角），做与训练一致的预处理（旋转与 padding 缩放）；
#     将图像与当前状态拼接为观测，调用远端策略推理服务获取一个动作序列（动作 chunk）；
#     以固定间隔（replan_steps）重复规划与执行，直到任务完成或达到最大步数；
#     保存每次回合（episode）的回放视频，并统计并打印成功率。

import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    """
    命令行参数（中文说明）
    字段含义：
    - host: 策略 WebSocket 服务的主机地址；
    - port: 策略 WebSocket 服务端口；
    - resize_size: 模型期望的输入图像尺寸（方形，预处理会保持纵横比并 pad）；
    - replan_steps: 每次推理得到一个动作序列后，仅执行前 replan_steps 步再重新规划；
    - task_suite_name: LIBERO 任务套件名称（libero_spatial/libero_object/libero_goal/libero_10/libero_90）；
    - num_steps_wait: 回合开始时的等待步数（让物体在仿真中稳定落下）；
    - num_trials_per_task: 每个任务评估的回合数；
    - video_out_path: 回放视频输出目录；
    - seed: 随机种子（影响初始化与仿真中的随机性）。
    """
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"   # 策略服务主机
    port: int = 8000    # 策略服务端口
    resize_size: int = 224  # 送入模型的图像边长（方形）
    replan_steps: int = 5   # 每次只执行这么多步后重新请求新动作序列

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    """
    在 LIBERO 任务套件上评估策略（中文说明）。
    参数：
        args: 'Args'，包含服务器地址端口、图像尺寸、重规划步数、任务套件名、等待步数、回合次数、视频输出路径与随机种子。
    过程：
    1. 设置随机种子；
    2. 加载任务套件与任务数；
    3. 为每个任务、每个回合：重置环境、设置初始状态、等待稳定、图像与状态预处理、请求远端策略动作序列并执行；
    4. 记录成功率并导出回放视频。

    返回：
        None，本函数以日志与文件输出形式报告结果。
    """

    # 设置随机种子，保证评估可复现
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    """
    构建并返回 LIBERO 环境与任务描述（中文说明）。
    参数：
    - task: 一个 LIBERO 任务对象，包含语言描述、问题目录与 bddl 文件名等；
    - resolution: 相机高度与宽度（方形分辨率），用于仿真环境渲染；
    - seed: 随机种子，用于环境初始化（会影响物体初始位置等随机因素）。
    返回：
    - (env, task_description):
        - env: 'OffScreenRenderEnv' 实例，可进行 'reset/step'；
        - task_description: 该任务的自然语言描述字符串。
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    """
    将四元数转换为轴角表示（中文说明）。
    参考来源: robosuite（链接见下）。
    参数：
    - quat: 长度为 4 的四元数数组 [x, y, z, w]，要求 w∈[-1, 1]。
    返回：
    - 长度为 3 的轴角向量（旋转轴乘以旋转角）。当旋转角接近 0 时，返回全零向量。
    引用实现:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f90see48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    # 截剪四元数的 w 分量，避免数值越界
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])  # 分母: sin(theta/2) 的绝对值
    if math.isclose(den, 0.0):  # 接近零角度旋转，直接返回零向量
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den  # 轴向量 * 角度


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)   # 使用 tyro 生成 CLI 并调用入口函数
