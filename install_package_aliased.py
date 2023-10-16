#!/usr/bin/env python3

import os
import re
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from shutil import copyfile, copytree, rmtree
from subprocess import check_call

import toml

if __name__ == "__main__":
    # parse arguments ######################################################################
    parser = ArgumentParser()
    parser.add_argument("new_name", help="The aliased name for this package when installing it.")
    parser.add_argument(
        "--install", help="Install the package if this flag is present.", action="store_true"
    )
    parser.add_argument(
        "--test", help="Test the (new named) package if this flag is present.", action="store_true"
    )
    args = parser.parse_args()
    new_name = args.new_name

    # validate the new name ###############################################################
    assert re.match("[-a-zA-Z0-9_]+", new_name) is not None, "Invalid package name"

    # create the new package description ##################################################
    root_path = Path(__file__).parent.absolute()
    current_config = toml.loads((root_path / "pyproject.toml").read_text())
    new_package_path = Path(__file__).parent.absolute() / new_name
    print(f"new_package_path: {new_package_path}")

    # remove the old generated package ####################################################
    if new_package_path.exists():
        rmtree(new_package_path)

    # copy files ##########################################################################
    for path in ["torch2jax", "pyproject.toml", "setup.py", "tests"]:
        src = root_path / path
        dest = new_package_path / path
        if src.is_file():
            copyfile(src, dest)
        else:
            copytree(src, dest)

    # rename the source directory #########################################################
    os.rename(new_package_path / "torch2jax", new_package_path / new_name)

    # write new config ####################################################################
    new_config = deepcopy(current_config)
    new_config["project"]["name"] = new_name
    Path(new_package_path / "pyproject.toml").write_text(toml.dumps(new_config))

    # rewrite the setup.py file
    setup_py = Path(root_path / "setup.py").read_text()
    new_setup_py = re.sub(
        f"name=\"{current_config['project']['name']}\"", f'name="{new_name}"', setup_py
    )
    Path(new_package_path / "setup.py").write_text(new_setup_py)

    # rename the test files ###############################################################
    test_files = sum(
        [
            [Path(root).absolute() / f for f in fnames if Path(f).suffix == ".py"]
            for (root, _, fnames) in os.walk(new_package_path / "tests")
        ],
        [],
    )
    for test_file in test_files:
        Path(test_file).write_text(
            re.sub(
                f"from {current_config['project']['name']}",
                f"from {new_name}",
                Path(test_file).read_text(),
            )
        )

    if args.install:
        # install the package ##################################################################
        check_call([sys.executable, "-m", "pip", "install", "-e", str(new_package_path)])

    if args.test:
        assert args.install, "You must install package before testing it, sorry."
        check_call([sys.executable, "-m", "pytest", str(new_package_path / "tests")])
