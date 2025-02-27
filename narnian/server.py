import torch
import base64
from flask_cors import CORS
from threading import Thread
from .environment import Environment
from flask import Flask, jsonify, render_template, request, send_from_directory


class Server:

    def __init__(self, env: Environment, root: str = 'viewer/www'):
        self.env = env
        self.env.using_server = True  # forcing
        self.root = root
        self.root_css = root + "/static/css"
        self.root_js = root + "/static/js"
        self.app = Flask(__name__, template_folder=self.root)
        CORS(self.app)  # to handle cross-origin requests (needed for development)
        self.register_routes()

        # starting a new thread
        thread = Thread(target=self.__run_server)
        thread.start()

    def __run_server(self):
        self.app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)  # Run Flask with threading enabled

    def register_routes(self):
        self.app.add_url_rule('/', view_func=self.serve_index, methods=['GET'])
        self.app.add_url_rule('/<path:filename>', view_func=self.serve_root, methods=['GET'])
        self.app.add_url_rule('/static/css/<path:filename>', view_func=self.serve_static_css, methods=['GET'])
        self.app.add_url_rule('/static/js/<path:filename>', view_func=self.serve_static_js, methods=['GET'])
        self.app.add_url_rule('/get_play_pause_status', view_func=self.get_play_pause_status, methods=['GET'])
        self.app.add_url_rule('/ask_to_pause', view_func=self.ask_to_pause, methods=['GET'])
        self.app.add_url_rule('/ask_to_play', view_func=self.ask_to_play, methods=['GET'])
        self.app.add_url_rule('/get_env_name', view_func=self.get_env_name, methods=['GET'])
        self.app.add_url_rule('/get_summary', view_func=self.get_summary, methods=['GET'])
        self.app.add_url_rule('/get_authority', view_func=self.get_authority, methods=['GET'])
        self.app.add_url_rule('/get_behav', view_func=self.get_behav, methods=['GET'])
        self.app.add_url_rule('/get_behav_status', view_func=self.get_behav_status, methods=['GET'])
        self.app.add_url_rule('/get_list_of_agents', view_func=self.get_list_of_agents, methods=['GET'])
        self.app.add_url_rule('/get_list_of_streams', view_func=self.get_list_of_streams, methods=['GET'])
        self.app.add_url_rule('/get_stream', view_func=self.get_stream, methods=['GET'])
        self.app.add_url_rule('/get_console', view_func=self.get_console, methods=['GET'])

    @staticmethod
    def pack_data(_data):
        _type = type(_data).__name__ if _data is not None else "none"

        def is_tensor_or_list_of_tensors(_d):
            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], torch.Tensor):
                return True
            elif isinstance(_d, torch.Tensor):
                return True
            else:
                return False

        # list of pytorch tensors (or nones)
        def encode_tensor_or_list_of_tensors(_d):
            _t = ""

            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], torch.Tensor):
                found_tensor = False
                _data64 = []
                for t in _d:
                    if t is not None:
                        if not found_tensor:
                            found_tensor = True
                            _t = "list_" + type(_d[0]).__name__ + "_" + _d[0].dtype.__str__().split('.')[-1]
                        _data64.append(base64.b64encode(t.detach().cpu().numpy().tobytes()).decode('utf-8'))
                    else:
                        _data64.append(None)
                if not found_tensor:
                    _t = "none"
                _d = _data64

                # pytorch tensor
            if isinstance(_d, torch.Tensor):
                _t = _d.dtype.__str__().split('.')[-1]
                _d = base64.b64encode(_d.detach().cpu().numpy()).decode('utf-8')

            return _d, _t

        if _type == "dict":
            keys = list(_data.keys())
            for k in keys:
                v = _data[k]
                if is_tensor_or_list_of_tensors(v):
                    d, t = encode_tensor_or_list_of_tensors(v)
                    del _data[k]
                    k = k + "-" + t
                    _data[k] = d
        else:
            if is_tensor_or_list_of_tensors(_data):
                _data, t = encode_tensor_or_list_of_tensors(_data)
                _type += "_" + t
            else:
                pass

        # generate JSON
        return jsonify({"data": _data, "type": _type})

    def serve_index(self):
        return send_from_directory(self.root, 'index.html')

    def serve_root(self, filename):
        return send_from_directory(self.root, filename)

    def serve_static_js(self, filename):
        return send_from_directory(self.root_js, filename)

    def serve_static_css(self, filename):
        return send_from_directory(self.root_css, filename)

    def get_play_pause_status(self):
        ret = {'status': None,
               'still_to_play': self.env.skip_clear_for}
        if self.env.step == self.env.steps:
            ret['status'] = 'ended'
        elif self.env.step_event.is_set():
            ret['status'] = 'playing'
        elif self.env.wait_event.is_set():
            ret['status'] = 'paused'
        return Server.pack_data(ret)

    def ask_to_play(self):
        steps = int(request.args.get('steps'))
        self.env.skip_clear_for = steps - 1
        self.env.step_event.set()
        return Server.pack_data(self.env.step)

    def ask_to_pause(self):
        self.env.skip_clear_for = 0
        return Server.pack_data(self.env.step)

    def get_env_name(self):
        return Server.pack_data(self.env.name)

    def get_summary(self):
        agent_name = request.args.get('agent_name')
        desc = str(self.env.agents[agent_name]) if agent_name != self.env.name else str(self.env)
        return Server.pack_data(desc)

    def get_authority(self):
        agent_name = request.args.get('agent_name')
        return Server.pack_data(self.env.agents[agent_name].authority)

    def get_behav(self):
        agent_name = request.args.get('agent_name')
        behav = self.env.agents[agent_name].behav if agent_name != self.env.name else self.env.behav
        return Server.pack_data(str(behav.to_graphviz().source))

    def get_behav_status(self):
        agent_name = request.args.get('agent_name')
        behav = self.env.agents[agent_name].behav if agent_name != self.env.name else self.env.behav
        state = behav.states[behav.state][2] if behav.state is not None else None
        action = behav.action[2] if behav.action is not None else None
        return Server.pack_data({'state': state, 'action': action})

    def get_list_of_agents(self):
        agent_name = request.args.get('agent_name')
        agents = self.env.agents[agent_name].known_agents if agent_name != self.env.name else self.env.agents
        return Server.pack_data(list(agents.keys()))

    def get_list_of_streams(self):
        agent_name = request.args.get('agent_name')
        streams = self.env.agents[agent_name].known_streams if agent_name != self.env.name else self.env.streams
        decoupled_streams = []
        for stream in streams.keys():
            decoupled_streams.append(stream + " [y]")
            decoupled_streams.append(stream + " [d]")
        return Server.pack_data(decoupled_streams)

    def get_stream(self):
        agent_name = request.args.get('agent_name')
        stream_name = request.args.get('stream_name')
        since_step = int(request.args.get('since_step'))
        data_id = 0
        if stream_name.endswith(" [y]"):
            stream_name = stream_name[0:stream_name.find(" [y]")]
            data_id = 0
        elif stream_name.endswith(" [d]"):
            stream_name = stream_name[0:stream_name.find(" [d]")]
            data_id = 1
        if agent_name != self.env.name:
            ks, data, last_k = self.env.agents[agent_name].known_streams[stream_name].get_since(since_step, data_id)
        else:
            ks, data, last_k = self.env.streams[stream_name].get_since(since_step, data_id)
        return Server.pack_data({
            "ks": ks,
            "data": data,
            "last_k": last_k
        })

    def get_console(self):
        agent_name = request.args.get('agent_name')
        agent = self.env.agents[agent_name] if agent_name != self.env.name else self.env
        return Server.pack_data({'output_messages': agent.output_messages,
                                 'output_messages_count': agent.output_messages_count,
                                 'output_messages_last_pos': agent.output_messages_last_pos,
                                 'output_messages_ids': agent.output_messages_ids})
