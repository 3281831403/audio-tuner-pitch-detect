# 吉他调音器：YIN 音高检测

## 题面

给吉他加调音器后，低音弦总被认成高八度、指针还一直抖：差分函数不除以窗长，幅度随帧长变化；累积均值归一化首项写成 0，后续全为 1；取第一个谷而不是第一个低于阈值的谷，倍频误判；插值用线性，音分误差 5；频率换算用 sample_rate/lag 但 lag 为浮点截断；音分公式用自然对数，偏差 0.7；音名计算用四舍五入到最近半音但基准错（A4=432）；静音帧不返回 -1，输出随机音高；批量共用差分缓冲，帧之间串；max_lag=0 不校验，除零。tuner/yin.py 是核心，tests/test_yin.py 15个红。要修的是：差分窗长归一化、累积均值首项、阈值谷选择、抛物线插值、浮点 lag 频率换算、以 2 为底的音分公式、A4=440 基准、静音返回 -1、批量缓冲隔离、max_lag 校验。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始基线必须恰好为 `15 failed`，且不应出现 collection error、import error 或 syntax error。完成实现后，同一命令应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 接口约定

`tuner/yin.py` 提供 8 个待实现函数和一个 `TunerError` 异常：

- `difference_fn(frame, max_lag)`：返回长度为 `max_lag` 的 YIN 差分函数，按窗长 `len(frame) - max_lag` 归一化。
- `cumulative_mean(diff)`：返回与输入等长的累积均值归一化结果，首项为 `1.0`。
- `pick_lag(cmnd, threshold)`：返回第一个足够深的谷对应的采样 lag，找不到时返回 `-1`。
- `parabolic_interp(cmnd, lag)`：用三点抛物线插值细化 lag，频率误差须在 1 音分内。
- `freq_from_lag(lag, sample_rate)`：按 `sample_rate / lag` 换算频率，`lag == 0` 抛 `TunerError`。
- `cents_offset(freq, target)`：返回相对目标音的偏移，公式为 `1200 * log2(freq / target)`。
- `nearest_note(freq)`：返回 `(音名, 八度)`，以 A4 = 440 Hz 为基准。
- `detect_batch(frames, config)`：批量检测并保持顺序；任一帧非法时整体回滚，不返回部分结果。

采样率单位为 Hz，帧长为采样点数，频率单位为 Hz；检测容差为 1 音分。音频一律使用 `list[float]`，静音帧以 `freq == -1.0` 表示。

## 约束

- `tests/test_yin.py` 不可修改，只修改实现。
- 仅使用 Python 标准库（`math`）与 `pytest`；离线运行，不使用随机数，不调用 `sleep`。
- 初始状态下 8 个函数全部 `raise NotImplementedError`，因此 15 个测试应全部失败。
