from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor
from farlog import getLogger

logger = getLogger("funtianchi.process")


def main() -> None:
    """根据命令行配置运行数据清洗任务。"""
    cfg = init_configs()
    if cfg.executor_type == "default":
        executor = DefaultExecutor(cfg)
    elif cfg.executor_type == "ray":
        from data_juicer.core.executor import RayExecutor

        executor = RayExecutor(cfg)
    else:
        raise ValueError(f"不支持的执行器类型: {cfg.executor_type}")
    try:
        executor.run()
    except Exception:
        logger.exception("数据处理失败")
        raise


if __name__ == "__main__":
    main()
