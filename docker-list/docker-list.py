#!/usr/bin/env python3

import docker

client = docker.from_env()
print(client.ping())
print(client.containers.list())
