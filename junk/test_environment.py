import sys

REQUIRED_PYTHON = "python3"
REQUIRED_MAJOR = 3


def main():
    if sys.version_info.major != REQUIRED_MAJOR:
        raise TypeError(f"This project requires Python {REQUIRED_MAJOR}. Found: Python {sys.version}")
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
