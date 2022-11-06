import json
import tornado.web
from tornado import gen 
import asyncio

from typing import Any
from tornado import websocket, web, ioloop, httputil
from quantumdraw.server.levels import potentials
from quantumdraw.server.scores import get_ai_score, get_user_score, get_solution

from quantumdraw.wavefunction.multiqubits_wavefunction import MultiQBitWaveFunction, GaussianAnsatz
from qiskit.circuit.library import RealAmplitudes
from copy import deepcopy
from qiskit.algorithms import optimizers 
import numpy as np

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

    async def on_message(self, message):

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

            self.ipot = decoded_message['data']
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

            # send the score and hint data
            hint, score = get_user_score(data, self.current_level_potential)
            self.write_message(json.dumps({'type': 'user_score', 'score': score, 'points': hint}))

            # define the wave function            
            domain = {'xmin': -5., 'xmax': 5.}
            ansatz = GaussianAnsatz(6, domain, self.ipot, parameters='add')
            wf = MultiQBitWaveFunction(self.current_level_potential, ansatz, domain, self.ipot, num_shots=10000)

            def objective_function(params):

                wf.params = params
                counts = wf.sample()
                return wf.nuclear_potential_count(counts) +  wf.kinetic_energy()
    

            def callback(res):
                counts = wf.sample().detach().numpy().tolist()
                counts = np.sqrt(counts)
                counts /= np.max(counts)
                points = list(zip(wf.xvect.numpy().tolist(), counts))
                score = wf.get_score()
                out_data = json.dumps({'type': 'ai_score', 'score': score, 'points': points})
                self.write_message(out_data)
                


            # Initialize the COBYLA optimizer
            optimizer = optimizers.COBYLA(maxiter=250, tol=0.0001,  callback=callback)


            # Create the initial parameters (noting that our single qubit variational form has 3 parameters)
            params = np.random.rand(wf.ansatz.num_parameters)
            params = 0.*np.ones(wf.ansatz.num_parameters)

            async def update_ai():
                res = optimizer.minimize(fun=objective_function, x0=params)
                sol = get_solution(self.current_level_potential)
                self.write_message(json.dumps({'type': 'game_over', 'points': sol}))
                
            async def update_user():
                data = decoded_message['data']
                # send the score and hint data
                hint, score = get_user_score(data, self.current_level_potential)
                self.write_message(json.dumps({'type': 'user_score', 'score': score, 'points': hint}))                

            if not self.ai_task:
                self.ai_task = await asyncio.create_task(update_ai())
                await asyncio.create_task(update_user())
                
               
                

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

