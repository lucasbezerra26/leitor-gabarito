import pytest

from school_test_reader import read_img


@pytest.fixture
def image_1():
    return "teste1.jpeg"


@pytest.fixture
def marked_expected():
    return {
        1: "a",
        2: "a",
        3: "e",
        4: None,
        5: None,
        6: "b",
        7: "a",
        8: "b",
        9: None,
        10: "a",
        11: "e",
        12: "b",
        13: "e",
        14: "b",
        15: "c",
    }


@pytest.fixture
def marked_detailed_expected():
    return {
        1: {'a': True, 'b': False, 'c': False, 'd': False, 'e': False},
        2: {'a': True, 'b': False, 'c': False, 'd': False, 'e': False},
        3: {'a': False, 'b': False, 'c': False, 'd': False, 'e': True},
        4: {'a': False, 'b': False, 'c': True, 'd': True, 'e': False},
        5: {'a': False, 'b': False, 'c': False, 'd': False, 'e': False},
        6: {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
        7: {'a': True, 'b': False, 'c': False, 'd': False, 'e': False},
        8: {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
        9: {'a': False, 'b': False, 'c': False, 'd': False, 'e': False},
        10: {'a': True, 'b': False, 'c': False, 'd': False, 'e': False},
        11: {'a': False, 'b': False, 'c': False, 'd': False, 'e': True},
        12: {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
        13: {'a': False, 'b': False, 'c': False, 'd': False, 'e': True},
        14: {'a': False, 'b': True, 'c': False, 'd': False, 'e': False},
        15: {'a': False, 'b': False, 'c': True, 'd': False, 'e': False}
    }


def test_read_img_return_tuple(image_1):
    response = read_img(image_1)
    assert isinstance(response, tuple)


def test_expected_answer(image_1, marked_expected):
    _, marked = read_img(image_1)
    assert marked == marked_expected


def test_expected_answer_detailed(image_1, marked_detailed_expected):
    detailed, _ = read_img(image_1)
    assert detailed == marked_detailed_expected
