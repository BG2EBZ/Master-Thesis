import sys

import test_env


def _strip_mode_args(argv):
    forwarded = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            skip_next = True
            continue
        if arg.startswith("--mode="):
            continue
        forwarded.append(arg)
    return forwarded


def main():
    forwarded = _strip_mode_args(sys.argv[1:])
    test_env.main(["--mode", "record", *forwarded])


if __name__ == "__main__":
    main()
