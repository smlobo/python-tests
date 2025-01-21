#!/usr/bin/env python3

import http.server
import constants

class HttpServer(http.server.HTTPServer):
    counter = 0

class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        country = "Unknown"
        if self.path == constants.ASIA_PATH:
            country = constants.ASIA_COUNTRY
        elif self.path == constants.AMERICA_PATH:
            country = constants.AMERICA_COUNTRY
        print(f'[{self.server.counter}] Received {self.path} request from: {self.headers["User-Agent"]}')
        self.server.counter += 1
        self.send_response(http.HTTPStatus.OK, country)
        self.send_header('Server', self.version_string())
        self.end_headers()
        self.wfile.write(bytes(country, 'utf-8'))

    def log_message(self, format, *args):
        pass


def run(server_class, handler_class):
    server_address = (constants.SERVER, constants.PORT)
    httpd = server_class(server_address, handler_class)
    print(f'Starting Python server on {server_address}')
    httpd.serve_forever()


if __name__ == '__main__':
    run(HttpServer, HTTPRequestHandler)
