import argparse
import os
from pathlib import Path

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.mono_sfm_loader import MonoSFMLoader
from gtsfm.runner.gtsfm_runner_base import GtsfmRunnerBase

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

logger = logger_utils.get_logger()


class GtsfmRunnerMonoSFM(GtsfmRunnerBase):
    tag = "GTSFM on Dataset in Olsson's Lund format"

    def construct_argparser(self) -> argparse.ArgumentParser:
        parser = super(GtsfmRunnerMonoSFM, self).construct_argparser()

        parser.add_argument("--dataset_root", type=str, required=True, help="")

        return parser

    def construct_loader(self) -> LoaderBase:
        loader = MonoSFMLoader(
            self.parsed_args.dataset_root,
            max_frame_lookahead=self.parsed_args.max_frame_lookahead,
            max_resolution=self.parsed_args.max_resolution,
        )

        return loader


if __name__ == "__main__":
    runner = GtsfmRunnerMonoSFM()
    runner.run()
