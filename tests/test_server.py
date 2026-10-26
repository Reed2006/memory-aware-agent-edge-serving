import socket

import pytest

from phase1.server import assert_port_free


def test_occupied_port_is_rejected_before_loading_another_model():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(RuntimeError, match='already in use'):
            assert_port_free(listener.getsockname()[1])


def test_free_port_can_be_reserved():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    assert_port_free(port)
