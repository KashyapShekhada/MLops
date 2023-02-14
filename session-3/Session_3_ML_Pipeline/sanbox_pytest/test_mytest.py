from mytest import square
import pytest

@pytest.fixture
def input_value():
    return 4

# def test_square_gives_correct_value():
#     subject = square(2)

#     assert subject == 4

def test_square_gives_correct_value(input_value):
    subject = square(input_value)

    assert subject == 16