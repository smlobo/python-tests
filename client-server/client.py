#!/usr/bin/env python3

import requests
import time
import constants


def do_get(path: str, count: int) -> int:
    url = 'http://' + constants.SERVER + ":" + str(constants.PORT) + path
    response = requests.get(url)
    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return count
    print(f'[{count}] Got: {response.status_code} {{{response.headers["Server"]}}}: {response.text}')
    count += 1
    time.sleep(2)
    return count


if __name__ == '__main__':
    count: int = 0
    while True:
        count = do_get(constants.ASIA_PATH, count)
        count = do_get(constants.AMERICA_PATH, count)
