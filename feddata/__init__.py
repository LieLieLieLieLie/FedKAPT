from config import Config


def get_data_module(cfg: Config):
    if cfg.data.dataset == "cwru":
        from feddata.cwru import CWRUDataModule
        return CWRUDataModule(cfg)
    if cfg.data.dataset == "wdbc":
        from feddata.wdbc import WDBCDataModule
        return WDBCDataModule(cfg)
    raise ValueError(
        f"Unknown dataset: {cfg.data.dataset}. Expected 'cwru' or 'wdbc'."
    )
