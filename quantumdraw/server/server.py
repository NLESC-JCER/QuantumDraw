import json
import tornado.web
import asyncio

from typing import Any
from tornado import websocket, web, ioloop, httputil
from quantumdraw.server.levels import potentials
from quantumdraw.server.scores import get_ai_score, get_user_score


class IndexHandler(web.RequestHandler):
    def get(self):
        self.render("index.html")


class SocketHandler(websocket.WebSocketHandler):
    ai_task = None
    current_level_potential = None

    def __init__(self, application: tornado.web.Application, request: httputil.HTTPServerRequest,
                 **kwargs: Any) -> None:
        super().__init__(application, request, **kwargs)

        self.ipot = 1
        self.current_level_potential = potentials[self.ipot]
        self.default_sleep_time = 0.5
        self.sleep_time = self.default_sleep_time

    def check_origin(self, origin):
        return True

    def on_message(self, message):
        decoded_message = json.loads(message)

        if decoded_message['type'] == 'ping':
            self.write_message(json.dumps({'type': 'pong'}))

        if decoded_message['type'] == 'speed':
            print('change speed')
            if self.sleep_time == self.default_sleep_time:
                self.sleep_time = 0.0
            else:
                self.sleep_time = self.default_sleep_time

        if decoded_message['type'] == 'reset':
            if self.ai_task:
                self.ai_task.cancel()
                self.ai_task = None

            self.current_level_potential = potentials[decoded_message['data']]
            
            #print(self.current_level_potential)
            # self.current_level_pot = None
            # if decoded_message['data']:
                
            num_samples = 50
            low_x = -5
            high_x = 5
            points = []
            for sample_num in range(0, num_samples):
                x = low_x + sample_num * (high_x - low_x) / num_samples
                y = self.current_level_potential(x)
                points.append([x, y])
            self.write_message(json.dumps({'type': 'potential', 'data': points}))

        if decoded_message['type'] == 'guess':
            data = decoded_message['data']
            score = get_user_score(data, self.current_level_potential)
            self.write_message(json.dumps({'type': 'user_score', 'score': score}))

            async def loop():
                iterator = get_ai_score(self.current_level_potential)
                for step in iterator:
                    points = step[0]
                    score = step[1]
                    data = json.dumps({'type': 'ai_score', 'score': score, 'points': points})
                    self.write_message(data)
                    await asyncio.sleep(self.sleep_time)

            if not self.ai_task:
                self.ai_task = asyncio.get_event_loop().create_task(loop())

        # print('message', decoded_message)

    def open(self):
        print('ws open')

    def on_close(self):
        print('ws close')


app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/(favicon.ico)', web.StaticFileHandler, {'path': '../'}),
    (r'/(rest_api_example.png)', web.StaticFileHandler, {'path': './'}),
])

if __name__ == '__main__':
    app.listen(8888)
    ioloop.IOLoop.instance().start()

