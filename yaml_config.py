# coding: UTF-8
import os

from fvcore.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(
            self,
            cfg_filename: str,
            need_version: str = None,
            allow_unsafe: bool = True,
    ):

        assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"

        loaded_cfg = _CfgNode.load_yaml_with_base(
            cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        if need_version is not None:
            load_version = loaded_cfg.get("VERSION", None)
            # 当load_version是None的时候表示无需版本号，忽略
            if need_version is not None and load_version is not None and need_version != load_version:
                raise TypeError(
                    "load version:{},need version:{}, config type:{}, file_name:{}"
                    .format(load_version, need_version, loaded_cfg, cfg_filename))
        self.clear()
        self.update(loaded_cfg)
        print(
            "[CfgNode.merge_from_file()] load success, need_version:{}, file_name:{}\n{}"
            .format(need_version, cfg_filename, loaded_cfg))

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        print("[CfgNode.dump()] args:{}, kwargs:{},".format(args, kwargs))
        return super().dump(*args, **kwargs)
