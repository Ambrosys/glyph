import os

def main():
    this_dir = os.path.dirname(__file__)

    header = """
name: py35
dependencies:
- python=3.5.1=0
- pip:
"""

    with open(os.path.join(this_dir, "../requirements.txt"), "r") as f:
        content = filter(lambda line: line[0] != "#", f.readlines())

    with open(os.path.join(this_dir, ".environment.yml"), "w") as f:
        f.writelines(header)
        f.writelines(map(lambda s: "  - "+s, content))


if __name__ == "__main__":
    main()
