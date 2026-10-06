"""Public dispatcher. Help requires only Python's standard library."""
import sys


def main():
    commands = ('chg2cap', 'qwen', 'freeze', 'prepare', 'extract', 'download-metrics')
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m captioning {chg2cap,qwen,freeze,prepare,extract,download-metrics} --help')
        return
    command = sys.argv.pop(1)
    if command not in commands:
        raise SystemExit('Unknown command: ' + command)
    if command == 'chg2cap':
        from captioning.cli import main as execute
    elif command == 'qwen':
        from captioning.qwen import main as execute
    elif command == 'freeze':
        from captioning.protocol import main as execute
    elif command == 'extract':
        from captioning.extract import main as execute
    elif command == 'download-metrics':
        from captioning.download import main as execute
    else:
        from captioning.prepare import main as execute
    execute()


if __name__ == '__main__':
    main()
