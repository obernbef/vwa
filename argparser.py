import argparse


def createparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, train and save a connect-four AI")

    parser.add_argument("pattern", metavar="pattern", help="pattern to be replaced", type=str)

    parser.add_argument("replacement",
                        metavar="replacement",
                        help="string to replace the selected part",
                        type=str)

    parser.add_argument("files",
                        metavar="files",
                        help="files to be renamed",
                        nargs="+",
                        type=lambda x: isvalidfile(parser, x))

    parser.add_argument("-r",
                        "--regex",
                        help="use if pattern is a Regex",
                        dest="regex",
                        action="store_true")

    parser.add_argument("-e",
                        "--enum",
                        help="use to add enumeration at the end of each file (default: `(#n)`)",
                        dest="enum",
                        action="store_true")

    parser.add_argument("--enum-type",
                        help="change type of enumeration. use `#n` to indicate the number",
                        default="(#n)",
                        dest="enum_type",
                        type=str)

    parser.add_argument("-m",
                        "--mod-ext",
                        help="use to also modify file extensions",
                        dest="mod_ext",
                        action="store_true")

    return parser
