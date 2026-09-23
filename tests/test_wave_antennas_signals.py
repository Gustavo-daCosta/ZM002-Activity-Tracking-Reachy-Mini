"""SIGTERM and SIGHUP must run the same cleanup as Ctrl+C: stopping the demo over SSH sends SIGTERM, and a
dropped SSH connection sends SIGHUP to the remote foreground process."""

import signal

import pytest

from robot.apps.wave_antennas import install_signal_handlers


def test_handler_raises_keyboard_interrupt():
    handler = install_signal_handlers()
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


def test_install_registers_the_handler_for_sigterm():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        handler = install_signal_handlers()
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_install_registers_the_handler_for_sighup():
    previous = signal.getsignal(signal.SIGHUP)
    try:
        handler = install_signal_handlers()
        assert signal.getsignal(signal.SIGHUP) is handler
    finally:
        signal.signal(signal.SIGHUP, previous)
