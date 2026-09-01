"""A display-free contract test for package and module entry points."""

import importlib
import tkinter

import pytest


def test_import_is_side_effect_free_and_exposes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_tk_root(*_args: object, **_kwargs: object) -> object:
        pytest.fail("Importing csv_analyzer must not create a Tk root window.")

    monkeypatch.setattr(tkinter, "Tk", unexpected_tk_root)

    package = importlib.import_module("csv_analyzer")
    entry_point = importlib.import_module("csv_analyzer.__main__")

    assert callable(package.main)
    assert callable(entry_point.main)
