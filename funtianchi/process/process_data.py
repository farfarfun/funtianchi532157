from farlog import getLogger

from data_juicer.config import init_configs
from data_juicer.core import Executor


logger = getLogger("funtianchi.process")


def main():
    cfg = init_configs()
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    try:
        executor.run()
    except Exception:
        logger.exception("数据处理失败")
        raise


if __name__ == '__main__':
    main()
