#!/usr/bin/env python3

import requests
import time
import sys
import random
import constants


def do_get(portList: list[int], path: str, count: int) -> int:
    port = random.choice(portList)
    url = 'http://' + constants.SERVER + ":" + str(port) + path
    response = requests.get(url)
    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return count
    print(f'[{count}] To: {port}; Got: {response.status_code} ' +
          f'{{{response.headers["Server"]}}}: {response.text}')
    count += 1
    time.sleep(2)
    return count


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print(f'{sys.argv[0]} [start-port[:end-port]]')
        print(f'\tdefault : {constants.PORT}:{constants.PORT}')
        exit(1)

    portList = [constants.PORT]
    if len(sys.argv) == 2:
        if ':' in sys.argv[1]:
            portRange = sys.argv[1].split(':')
            assert len(portRange) == 2
            portStart = int(portRange[0])
            portEnd = int(portRange[1])
            portList = []
            while portStart <= portEnd:
                portList.append(portStart)
                portStart += 1
            print(f'{portList}')
        else:
            portList = [int(sys.argv[1])]

    count: int = 0
    while True:
        count = do_get(portList, constants.ASIA_PATH, count)
        count = do_get(portList, constants.AMERICA_PATH, count)
