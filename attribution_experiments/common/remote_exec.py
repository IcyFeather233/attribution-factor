"""ReCreate's environment inspection shim, restricted to registered containers."""
import sys
from .remote_runtime import RemoteRuntime


def main():
    args = sys.argv[1:]
    if not args or args.pop(0) != 'exec':
        raise SystemExit('Only exec is supported')
    workdir = '/testbed'
    if args[:1] == ['-w']:
        _, workdir = args[:2]
        args = args[2:]
    if len(args) < 2:
        raise SystemExit('Expected container and command')
    runtime = RemoteRuntime.attach(args[0])
    result = runtime.run(['run', '--nobanner', '--workdir=' + workdir, *args], timeout=100)
    print(result.stdout, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
