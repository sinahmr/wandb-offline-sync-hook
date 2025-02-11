from __future__ import annotations

import os
import subprocess
import time
from os import PathLike
from pathlib import Path

from wandb_osh import __version__
from wandb_osh.config import _command_dir_default
from wandb_osh.util.log import logger


class WandbSyncer:
    def __init__(
        self,
        command_dir: PathLike = _command_dir_default,
        wait: int = 1,
        wandb_options: list[str] | None = None,
        *,
        timeout: int | float = 600,
    ):
        """Class for interpreting command files and triggering
        `wandb sync`.

        Args:
            command_dir: Directory used for communication
            wait: Minimal time to wait before scanning command dir again
            wandb_options: Options to pass on to wandb
            timeout: Timeout for wandb sync. If <=0, no timeout.
        """
        if wandb_options is None:
            wandb_options = []
        self.command_dir = Path(command_dir)
        self.wait = wait
        self.wandb_options = wandb_options
        self._timeout = timeout
        self.disable_append = False
        logger.info(f"Timeout is set to {timeout}")

    def sync(self, dir: PathLike, append: bool = False) -> None:
        """Sync a directory. Thin wrapper around the `sync_dir` function.

        Args:
            dir: Directory with wandb files to be synced
            append: Use --append switch in wandb sync
        """
        sync_dir(dir, options=self.wandb_options + ['--append'] if append else [], timeout=self._timeout)

    def loop(self) -> None:
        """Read command files and trigger syncing"""
        logger.info(
            "wandb-osh v%s, starting to watch %s", __version__, self.command_dir
        )
        try:
            while True:
                start_time = time.time()
                self.command_dir.mkdir(parents=True, exist_ok=True)

                # sbatch command
                for file in self.command_dir.glob("*.sbatch"):
                    time.sleep(60)
                    try:
                        cmd = file.read_text().strip()
                        cmd = cmd.split(' ')
                        logger.info(f"Submitting sbatch job: {cmd}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            print(f"Job rescheduled successfully: {result.stdout}")
                        else:
                            logger.warning(f"Error while scheduling the job: {result.stderr}")
                        time.sleep(0.25)
                        if file.is_file():
                            file.unlink()
                    except subprocess.TimeoutExpired:
                        logger.warning("Timed out, trying later")

                for command_file in self.command_dir.glob("*.command"):
                    time.sleep(60)
                    target = Path(command_file.read_text().strip())
                    if not target.is_dir():
                        logger.error(
                            "Command file %s points to non-existing directory %s",
                            command_file,
                            target,
                        )
                        if command_file.is_file():
                            command_file.unlink()
                        continue

                    wandb_id = target.as_posix().split('-')[-1]
                    new = wandb_id not in self.get_seen()
                    logger.info(f"Syncing {target}{'' if new else ' using --append'} ...")
                    try:
                        append = False if self.disable_append else not new
                        self.sync(target, append=append)
                        if new:
                            self.append_to_seen(wandb_id)
                        time.sleep(0.25)
                        if command_file.is_file():
                            command_file.unlink()
                    except subprocess.TimeoutExpired:
                        logger.warning("Syncing %s timed out. Trying later.", target)
                        # from wandb_osh.hooks import TriggerWandbSyncHook
                        # TriggerWandbSyncHook(self.command_dir)(target)
                    print()

                # time.sleep(0.25)
                # for cf in sbatch_files:
                #     if cf.is_file():
                #         cf.unlink()
                ###

                if "PYTEST_CURRENT_TEST" in os.environ:
                    break
                time.sleep(max(0.0, (time.time() - start_time) - self.wait))
        except KeyboardInterrupt:
            pass
        except:
            subprocess.run(["sbatch", "--job-name=osh-crashed", "/home/sinah/workspace/ViT/drac/notification.sh"], capture_output=True, text=True)

    def get_seen(self):
        seen_file = self.command_dir / "seen.txt"
        if seen_file.exists():
            return set([wandb_id.strip() for wandb_id in seen_file.read_text().strip().split()])
        return set()

    def append_to_seen(self, wandb_id):
        seen_file = self.command_dir / "seen.txt"
        with seen_file.open("a") as f:
            f.write(wandb_id + "\n")



def sync_dir(
    dir: PathLike, options: list[str] | None = None, *, timeout: int | float = 0
) -> None:
    """Call wandb sync on a directory.

    Args:
        dir: Directory with wandb runs
        options: List of options to pass on to `wandb sync`
        timeout: Timeout for wandb sync. If <=0: no timeout
    """
    if options is None:
        options = []
    dir = Path(dir)
    command = ["wandb", "sync", *options, "."]
    if "PYTEST_CURRENT_TEST" in os.environ:
        logger.debug("Testing mode enabled. Not actually calling wandb.")
        logger.debug("Command would be: %s in %s", " ".join(command), dir)
        return
    _timeout = None if timeout <= 0 else timeout
    subprocess.run(command, cwd=dir, timeout=_timeout)
